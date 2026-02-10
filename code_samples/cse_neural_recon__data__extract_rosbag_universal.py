import os
import argparse
import json
import numpy as np
import cv2
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

def extract_bag(bag_path, output_root):
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    output_dir = os.path.join(output_root, bag_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Start with standard types
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f"\nProcessing: {bag_name}")
    
    # Detect robot prefix from bag topics (carter1, carter2, carter3, etc.)
    with Reader(bag_path) as reader:
        robot_prefix = None
        for c in reader.connections:
            topic = c.topic.lstrip('/')
            if topic.startswith('carter'):
                robot_prefix = topic.split('/')[0]
                break
        if robot_prefix is None:
            print(f"  [WARN] Could not detect robot prefix, skipping bag")
            return
        print(f"  [INFO] Detected robot prefix: {robot_prefix}")

    # TARGET CONFIGURATION - dynamically set based on robot prefix
    target_groups = {
        'info':  [f'{robot_prefix}/camera_info_left', f'{robot_prefix}/camera_info_right'],
        'pose':  [f'{robot_prefix}/gt_pose'],
        'rgb':   [f'{robot_prefix}/rgb_left/compressed', f'{robot_prefix}/rgb_right/compressed'],
        'depth': [f'{robot_prefix}/depth_left', f'{robot_prefix}/depth_right']
    }
    
    # Output file setup
    saved_intrinsics = set()
    pose_path = os.path.join(output_dir, "ground_truth.txt")
    pose_file = open(pose_path, 'w')
    pose_file.write("# timestamp tx ty tz qx qy qz qw\n")

    with Reader(bag_path) as reader:
        # --- DYNAMIC TYPE REGISTRATION ---
        # Iterate all connections first to register any unknown types found in this bag
        print("  [INIT] Registering types from bag...")
        for connection in reader.connections:
            if connection.msgtype not in typestore.types:
                try:
                    # Parse the raw definition text stored in the bag
                    # This teaches the store what "sensor_msgs/msg/CameraInfo" actually looks like
                    new_types = get_types_from_msg(connection.msgdef, connection.msgtype)
                    typestore.register(new_types)
                except Exception as e:
                    # print(f"    Warning: Could not register {connection.msgtype}: {e}")
                    pass

        # --- FILTERING ---
        connections = []
        for c in reader.connections:
            c_clean = c.topic.lstrip('/')
            for group, topics in target_groups.items():
                if c_clean in topics:
                    connections.append(c)
                    break
        
        total_msgs = sum(c.msgcount for c in connections)
        print(f"  [EXTRACT] Extracting {total_msgs} messages...")
        
        with tqdm(total=total_msgs) as pbar:
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                topic_clean = connection.topic.lstrip('/')
                
                try:
                    # Deserialize using the bag's OWN type name
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    ts_str = str(timestamp)
                    
                    # --- 1. CAMERA INFO (Intrinsics) ---
                    if topic_clean in target_groups['info']:
                        if topic_clean not in saved_intrinsics:
                            # Note: Fields are uppercase K, D in ROS CameraInfo
                            width = getattr(msg, 'width', 0)
                            height = getattr(msg, 'height', 0)
                            K = getattr(msg, 'K', [])
                            D = getattr(msg, 'D', [])
                            model = getattr(msg, 'distortion_model', 'unknown')

                            # Handle numpy arrays if present
                            if hasattr(K, 'tolist'): K = K.tolist()
                            if hasattr(D, 'tolist'): D = D.tolist()

                            intrinsics = {
                                "width": width, "height": height,
                                "K": K, "D": D, "model": model
                            }
                            
                            fname = topic_clean.replace(f'{robot_prefix}/', '').replace('/', '_')
                            info_path = os.path.join(output_dir, f"{fname}_intrinsics.json")
                            with open(info_path, 'w') as f:
                                json.dump(intrinsics, f, indent=4)
                            
                            saved_intrinsics.add(topic_clean)
                        
                        pbar.update(1)
                        continue

                    # --- 2. POSE ---
                    if topic_clean in target_groups['pose']:
                        # Traverse standard ROS structure
                        # msg.pose (PoseWithCovariance) -> .pose (Pose) -> .position/.orientation
                        # Or msg.pose (Pose) directly
                        p_obj = msg.pose
                        if hasattr(p_obj, 'pose'): # It's likely Odometry or CovarianceStamped
                            p_obj = p_obj.pose
                        
                        pos = p_obj.position
                        ori = p_obj.orientation
                        
                        ts_sec = timestamp / 1e9
                        line = f"{ts_sec:.6f} {pos.x} {pos.y} {pos.z} {ori.x} {ori.y} {ori.z} {ori.w}\n"
                        pose_file.write(line)
                        pbar.update(1)
                        continue

                    # --- 3. IMAGES ---
                    subfolder = topic_clean.replace(f'{robot_prefix}/', '').replace('/', '_')
                    save_folder = os.path.join(output_dir, subfolder)
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, f"{ts_str}.png")
                    
                    img_mat = None

                    # RGB
                    if topic_clean in target_groups['rgb']:
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        img_mat = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

                    # Depth
                    elif topic_clean in target_groups['depth']:
                        dtype = np.uint8
                        channels = 1
                        encoding = getattr(msg, 'encoding', '').lower()
                        
                        if "32fc1" in encoding:
                            raw_data = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                            img_mat = (raw_data * 1000).astype(np.uint16)
                        elif "16uc1" in encoding or "mono16" in encoding:
                            img_mat = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                        elif "mono8" in encoding:
                            img_mat = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)

                    if img_mat is not None:
                        cv2.imwrite(save_path, img_mat)

                except Exception as e:
                    print(f"\n[ERROR] Failed on {connection.topic} ({connection.msgtype}): {e}")
                    # raise e # Uncomment to see full traceback if needed
                    pass
                
                pbar.update(1)

    pose_file.close()
    print(f"Done. Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CSE Dataset (Robust Type Registration)")
    parser.add_argument("--root", type=str, required=True, help="Folder containing .bag files")
    parser.add_argument("--dest", type=str, required=True, help="Output folder")
    parser.add_argument("--single", action="store_true", help="Only process the first bag found")
    
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print(f"Error: Root path '{args.root}' does not exist.")
        exit()

    bag_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith('.bag')]
    bag_files.sort()

    if args.single:
        extract_bag(bag_files[0], args.dest)
    else:
        for bag in bag_files:
            extract_bag(bag, args.dest)
