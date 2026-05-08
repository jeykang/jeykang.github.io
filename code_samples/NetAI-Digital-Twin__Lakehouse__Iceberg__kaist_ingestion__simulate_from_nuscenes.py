#!/usr/bin/env python3
"""
KAIST Dataset Simulator

Generates synthetic KAIST-format data from the nuScenes mini dataset.
This allows testing the KAIST ingestion pipeline before the actual dataset arrives.

Mapping Strategy:
    nuScenes              →  KAIST
    ─────────────────────────────────────────
    scene                 →  Session (1 scene = 1 session)
    scene                 →  Clip (1 scene = 1 clip, could split if needed)
    sample                →  Frame
    sample_data (camera)  →  Camera
    sample_data (lidar)   →  Lidar  
    sample_data (radar)   →  Radar
    calibrated_sensor     →  Calibration
    sample_annotation     →  DynamicObject
    category              →  Category
    ego_pose              →  EgoMotion
    log + map             →  HDMap
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class KAISTSession:
    session_id: str
    session_name: str
    clip_id_list: List[str]


@dataclass
class KAISTClip:
    clip_id: str
    session_id: str
    clip_idx: int
    frame_id_list: List[str]
    date: str


@dataclass
class KAISTFrame:
    frame_id: str
    clip_id: str
    frame_idx: int
    sensor_timestamps: List[int]


@dataclass
class KAISTCalibration:
    clip_id: str
    sensor_name: str
    extrinsics: Dict[str, List[float]]  # {translation, rotation}
    camera_intrinsics: Optional[List[float]]


@dataclass
class KAISTCamera:
    frame_id: str
    clip_id: str
    system_timestamp: int
    sensor_timestamp: int
    camera_name: str
    filename: str


@dataclass
class KAISTLidar:
    frame_id: str
    clip_id: str
    system_timestamp: int
    sensor_timestamp: int
    filename: str


@dataclass
class KAISTRadar:
    frame_id: str
    clip_id: str
    system_timestamp: int
    sensor_timestamp: int
    radar_name: str
    filename: str


@dataclass
class KAISTCategory:
    category: str


@dataclass
class KAISTDynamicObject:
    frame_id: str
    clip_id: str
    boxes_3d: List[float]  # Flattened: [cx, cy, cz, l, w, h, yaw, ...]
    category: str


@dataclass
class KAISTEgoMotion:
    frame_id: str
    clip_id: str
    session_id: str
    translation: Dict[str, float]  # {x, y, z}
    rotation: Dict[str, float]     # {qw, qx, qy, qz}


@dataclass
class KAISTSessionEgoMotion:
    session_id: str
    translation: Dict[str, float]
    rotation: Dict[str, float]
    start: Dict[str, float]
    goal: Dict[str, float]


@dataclass
class KAISTHDMap:
    clip_id: str
    filename: str
    city: str
    site: str


class NuScenesToKAISTConverter:
    """
    Converts nuScenes mini dataset to KAIST format.
    """
    
    # Sensor name mappings
    CAMERA_SENSORS = {
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    }
    LIDAR_SENSORS = {"LIDAR_TOP"}
    RADAR_SENSORS = {
        "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
    }
    
    def __init__(self, nuscenes_root: Path, output_root: Path):
        self.nuscenes_root = Path(nuscenes_root)
        self.output_root = Path(output_root)
        
        # Find the JSON directory
        self.json_dir = self._find_json_dir()
        
        # Load nuScenes tables
        self.scenes = self._load_json("scene.json")
        self.samples = self._load_json("sample.json")
        self.sample_data = self._load_json("sample_data.json")
        self.calibrated_sensors = self._load_json("calibrated_sensor.json")
        self.sensors = self._load_json("sensor.json")
        self.ego_poses = self._load_json("ego_pose.json")
        self.sample_annotations = self._load_json("sample_annotation.json")
        self.instances = self._load_json("instance.json")
        self.categories = self._load_json("category.json")
        self.logs = self._load_json("log.json")
        self.maps = self._load_json("map.json")
        
        # Build lookup indices
        self._build_indices()
        
        # Output data structures
        self.kaist_sessions: List[KAISTSession] = []
        self.kaist_clips: List[KAISTClip] = []
        self.kaist_frames: List[KAISTFrame] = []
        self.kaist_calibrations: List[KAISTCalibration] = []
        self.kaist_cameras: List[KAISTCamera] = []
        self.kaist_lidars: List[KAISTLidar] = []
        self.kaist_radars: List[KAISTRadar] = []
        self.kaist_categories: List[KAISTCategory] = []
        self.kaist_dynamic_objects: List[KAISTDynamicObject] = []
        self.kaist_ego_motions: List[KAISTEgoMotion] = []
        self.kaist_session_ego_motions: List[KAISTSessionEgoMotion] = []
        self.kaist_hdmaps: List[KAISTHDMap] = []
        
    def _find_json_dir(self) -> Path:
        """Find the directory containing nuScenes JSON files."""
        candidates = [
            self.nuscenes_root / "v1.0-mini",
            self.nuscenes_root / "v1.0-mini" / "v1.0-mini",
            self.nuscenes_root,
        ]
        for candidate in candidates:
            if (candidate / "scene.json").exists():
                return candidate
        raise FileNotFoundError(f"Could not find nuScenes JSON files in {self.nuscenes_root}")
    
    def _load_json(self, filename: str) -> List[Dict]:
        """Load a JSON file from the nuScenes directory."""
        path = self.json_dir / filename
        if not path.exists():
            print(f"Warning: {filename} not found, using empty list")
            return []
        with open(path) as f:
            return json.load(f)
    
    def _build_indices(self):
        """Build lookup dictionaries for faster access."""
        # Token → record lookups
        self.sample_by_token = {s["token"]: s for s in self.samples}
        self.sensor_by_token = {s["token"]: s for s in self.sensors}
        self.calib_by_token = {c["token"]: c for c in self.calibrated_sensors}
        self.ego_pose_by_token = {e["token"]: e for e in self.ego_poses}
        self.instance_by_token = {i["token"]: i for i in self.instances}
        self.category_by_token = {c["token"]: c for c in self.categories}
        self.log_by_token = {l["token"]: l for l in self.logs}
        self.map_by_token = {m["token"]: m for m in self.maps}
        
        # Scene → samples mapping
        self.samples_by_scene: Dict[str, List[Dict]] = {}
        for sample in self.samples:
            scene_token = sample["scene_token"]
            if scene_token not in self.samples_by_scene:
                self.samples_by_scene[scene_token] = []
            self.samples_by_scene[scene_token].append(sample)
        
        # Sort samples by timestamp within each scene
        for scene_token in self.samples_by_scene:
            self.samples_by_scene[scene_token].sort(key=lambda s: s["timestamp"])
        
        # Sample → sample_data mapping
        self.sample_data_by_sample: Dict[str, List[Dict]] = {}
        for sd in self.sample_data:
            sample_token = sd["sample_token"]
            if sample_token not in self.sample_data_by_sample:
                self.sample_data_by_sample[sample_token] = []
            self.sample_data_by_sample[sample_token].append(sd)
        
        # Sample → annotations mapping
        self.annotations_by_sample: Dict[str, List[Dict]] = {}
        for ann in self.sample_annotations:
            sample_token = ann["sample_token"]
            if sample_token not in self.annotations_by_sample:
                self.annotations_by_sample[sample_token] = []
            self.annotations_by_sample[sample_token].append(ann)
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}{uuid.uuid4().hex[:24]}"
    
    def _get_sensor_channel(self, calib_token: str) -> str:
        """Get the sensor channel name from a calibrated_sensor token."""
        calib = self.calib_by_token.get(calib_token, {})
        sensor_token = calib.get("sensor_token", "")
        sensor = self.sensor_by_token.get(sensor_token, {})
        return sensor.get("channel", "UNKNOWN")
    
    def _quaternion_to_dict(self, quat: List[float]) -> Dict[str, float]:
        """Convert [w, x, y, z] quaternion list to dict."""
        if len(quat) >= 4:
            return {"qw": quat[0], "qx": quat[1], "qy": quat[2], "qz": quat[3]}
        return {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
    
    def _translation_to_dict(self, trans: List[float]) -> Dict[str, float]:
        """Convert [x, y, z] translation list to dict."""
        if len(trans) >= 3:
            return {"x": trans[0], "y": trans[1], "z": trans[2]}
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    def _convert_categories(self):
        """Convert nuScenes categories to KAIST format."""
        seen = set()
        for cat in self.categories:
            cat_name = cat["name"]
            if cat_name not in seen:
                self.kaist_categories.append(KAISTCategory(category=cat_name))
                seen.add(cat_name)
    
    def _convert_scene(self, scene: Dict) -> tuple:
        """
        Convert a nuScenes scene to KAIST Session + Clip.
        
        Returns (session_id, clip_id, frame_ids)
        """
        session_id = self._generate_id("sess_")
        clip_id = self._generate_id("clip_")
        
        # Get samples for this scene
        samples = self.samples_by_scene.get(scene["token"], [])
        frame_ids = []
        
        # Convert each sample to a Frame
        for frame_idx, sample in enumerate(samples):
            frame_id = self._generate_id("frame_")
            frame_ids.append(frame_id)
            
            # Collect sensor timestamps for this frame
            sensor_data = self.sample_data_by_sample.get(sample["token"], [])
            sensor_timestamps = [sd["timestamp"] for sd in sensor_data]
            
            self.kaist_frames.append(KAISTFrame(
                frame_id=frame_id,
                clip_id=clip_id,
                frame_idx=frame_idx,
                sensor_timestamps=sensor_timestamps,
            ))
            
            # Convert sensor data
            self._convert_sensor_data(frame_id, clip_id, session_id, sample, sensor_data)
            
            # Convert annotations for this frame
            self._convert_annotations(frame_id, clip_id, sample)
            
            # Convert ego pose
            self._convert_ego_motion(frame_id, clip_id, session_id, sample, sensor_data)
        
        # Create Session
        self.kaist_sessions.append(KAISTSession(
            session_id=session_id,
            session_name=scene.get("name", "unnamed"),
            clip_id_list=[clip_id],
        ))
        
        # Create Clip
        log = self.log_by_token.get(scene.get("log_token", ""), {})
        date_captured = log.get("date_captured", "2026-01-01")
        
        self.kaist_clips.append(KAISTClip(
            clip_id=clip_id,
            session_id=session_id,
            clip_idx=0,
            frame_id_list=frame_ids,
            date=date_captured,
        ))
        
        # Convert calibration (per clip)
        self._convert_calibration(clip_id)
        
        # Convert HDMap
        self._convert_hdmap(clip_id, scene)
        
        # Create session-level ego motion summary
        self._create_session_ego_motion(session_id, samples)
        
        return session_id, clip_id, frame_ids
    
    def _convert_sensor_data(
        self, 
        frame_id: str, 
        clip_id: str, 
        session_id: str,
        sample: Dict, 
        sensor_data: List[Dict]
    ):
        """Convert sample_data entries to Camera/Lidar/Radar tables."""
        for sd in sensor_data:
            channel = self._get_sensor_channel(sd["calibrated_sensor_token"])
            timestamp = sd["timestamp"]
            filename = sd["filename"]
            
            if channel in self.CAMERA_SENSORS:
                self.kaist_cameras.append(KAISTCamera(
                    frame_id=frame_id,
                    clip_id=clip_id,
                    system_timestamp=sample["timestamp"],
                    sensor_timestamp=timestamp,
                    camera_name=channel,
                    filename=filename,
                ))
            elif channel in self.LIDAR_SENSORS:
                self.kaist_lidars.append(KAISTLidar(
                    frame_id=frame_id,
                    clip_id=clip_id,
                    system_timestamp=sample["timestamp"],
                    sensor_timestamp=timestamp,
                    filename=filename,
                ))
            elif channel in self.RADAR_SENSORS:
                self.kaist_radars.append(KAISTRadar(
                    frame_id=frame_id,
                    clip_id=clip_id,
                    system_timestamp=sample["timestamp"],
                    sensor_timestamp=timestamp,
                    radar_name=channel,
                    filename=filename,
                ))
    
    def _convert_calibration(self, clip_id: str):
        """Convert calibrated_sensor entries to KAIST Calibration."""
        for calib in self.calibrated_sensors:
            channel = self._get_sensor_channel(calib["token"])
            
            extrinsics = {
                "translation": calib.get("translation", [0, 0, 0]),
                "rotation": calib.get("rotation", [1, 0, 0, 0]),
            }
            
            # Camera intrinsics (only for cameras)
            intrinsics = calib.get("camera_intrinsic", None)
            if intrinsics and len(intrinsics) > 0:
                # Flatten 3x3 matrix
                flat_intrinsics = []
                for row in intrinsics:
                    flat_intrinsics.extend(row)
            else:
                flat_intrinsics = None
            
            self.kaist_calibrations.append(KAISTCalibration(
                clip_id=clip_id,
                sensor_name=channel,
                extrinsics=extrinsics,
                camera_intrinsics=flat_intrinsics,
            ))
    
    def _convert_annotations(self, frame_id: str, clip_id: str, sample: Dict):
        """Convert sample_annotations to DynamicObject entries."""
        annotations = self.annotations_by_sample.get(sample["token"], [])
        
        for ann in annotations:
            # Get category name
            instance = self.instance_by_token.get(ann.get("instance_token", ""), {})
            category_token = instance.get("category_token", "")
            category = self.category_by_token.get(category_token, {})
            category_name = category.get("name", "unknown")
            
            # Build boxes_3d: [cx, cy, cz, l, w, h, yaw]
            # nuScenes uses [w, l, h] but we'll keep consistent ordering
            translation = ann.get("translation", [0, 0, 0])
            size = ann.get("size", [1, 1, 1])  # [w, l, h]
            rotation = ann.get("rotation", [1, 0, 0, 0])  # quaternion
            
            # Convert quaternion to yaw (simplified - just use qw for now)
            # In practice, you'd compute actual yaw from quaternion
            import math
            qw, qx, qy, qz = rotation[0], rotation[1], rotation[2], rotation[3]
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            
            boxes_3d = [
                translation[0],  # cx
                translation[1],  # cy
                translation[2],  # cz
                size[1],         # length (nuScenes uses [w, l, h])
                size[0],         # width
                size[2],         # height
                yaw,             # yaw angle
            ]
            
            self.kaist_dynamic_objects.append(KAISTDynamicObject(
                frame_id=frame_id,
                clip_id=clip_id,
                boxes_3d=boxes_3d,
                category=category_name,
            ))
    
    def _convert_ego_motion(
        self, 
        frame_id: str, 
        clip_id: str, 
        session_id: str,
        sample: Dict,
        sensor_data: List[Dict]
    ):
        """Convert ego_pose to EgoMotion."""
        # Use the first available ego_pose from sensor_data
        for sd in sensor_data:
            ego_pose_token = sd.get("ego_pose_token", "")
            ego_pose = self.ego_pose_by_token.get(ego_pose_token)
            
            if ego_pose:
                self.kaist_ego_motions.append(KAISTEgoMotion(
                    frame_id=frame_id,
                    clip_id=clip_id,
                    session_id=session_id,
                    translation=self._translation_to_dict(ego_pose.get("translation", [0, 0, 0])),
                    rotation=self._quaternion_to_dict(ego_pose.get("rotation", [1, 0, 0, 0])),
                ))
                break  # Only need one ego pose per frame
    
    def _create_session_ego_motion(self, session_id: str, samples: List[Dict]):
        """Create session-level ego motion summary (start/goal positions)."""
        if not samples:
            return
        
        # Get ego pose for first and last sample
        first_sample = samples[0]
        last_sample = samples[-1]
        
        start_pose = None
        goal_pose = None
        
        # Find ego poses
        for sd in self.sample_data_by_sample.get(first_sample["token"], []):
            ego_pose = self.ego_pose_by_token.get(sd.get("ego_pose_token", ""))
            if ego_pose:
                start_pose = ego_pose
                break
        
        for sd in self.sample_data_by_sample.get(last_sample["token"], []):
            ego_pose = self.ego_pose_by_token.get(sd.get("ego_pose_token", ""))
            if ego_pose:
                goal_pose = ego_pose
                break
        
        if start_pose and goal_pose:
            self.kaist_session_ego_motions.append(KAISTSessionEgoMotion(
                session_id=session_id,
                translation=self._translation_to_dict(start_pose.get("translation", [0, 0, 0])),
                rotation=self._quaternion_to_dict(start_pose.get("rotation", [1, 0, 0, 0])),
                start=self._translation_to_dict(start_pose.get("translation", [0, 0, 0])),
                goal=self._translation_to_dict(goal_pose.get("translation", [0, 0, 0])),
            ))
    
    def _convert_hdmap(self, clip_id: str, scene: Dict):
        """Convert log/map references to HDMap."""
        log = self.log_by_token.get(scene.get("log_token", ""), {})
        
        # nuScenes uses log.map_token, but maps are city-wide
        location = log.get("location", "unknown")
        
        # Parse location (e.g., "singapore-onenorth" -> city="singapore", site="onenorth")
        parts = location.split("-") if location else ["unknown", "unknown"]
        city = parts[0] if len(parts) > 0 else "unknown"
        site = parts[1] if len(parts) > 1 else "default"
        
        self.kaist_hdmaps.append(KAISTHDMap(
            clip_id=clip_id,
            filename=f"maps/{location}.json",
            city=city,
            site=site,
        ))
    
    def convert(self):
        """Run the full conversion."""
        print(f"Converting nuScenes data from: {self.json_dir}")
        print(f"Output directory: {self.output_root}")
        
        # Convert categories first
        self._convert_categories()
        print(f"  Converted {len(self.kaist_categories)} categories")
        
        # Convert each scene
        for scene in self.scenes:
            session_id, clip_id, frame_ids = self._convert_scene(scene)
            print(f"  Converted scene '{scene.get('name', 'unknown')}' → {len(frame_ids)} frames")
        
        print(f"\nConversion complete:")
        print(f"  Sessions:       {len(self.kaist_sessions)}")
        print(f"  Clips:          {len(self.kaist_clips)}")
        print(f"  Frames:         {len(self.kaist_frames)}")
        print(f"  Cameras:        {len(self.kaist_cameras)}")
        print(f"  Lidars:         {len(self.kaist_lidars)}")
        print(f"  Radars:         {len(self.kaist_radars)}")
        print(f"  Calibrations:   {len(self.kaist_calibrations)}")
        print(f"  DynamicObjects: {len(self.kaist_dynamic_objects)}")
        print(f"  EgoMotions:     {len(self.kaist_ego_motions)}")
        print(f"  HDMaps:         {len(self.kaist_hdmaps)}")
    
    def _to_dict_list(self, items: List) -> List[Dict]:
        """Convert dataclass list to dict list for JSON serialization."""
        return [asdict(item) for item in items]
    
    def save(self):
        """Save all KAIST-format JSON files."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        outputs = {
            "session.json": self.kaist_sessions,
            "clip.json": self.kaist_clips,
            "frame.json": self.kaist_frames,
            "calibration.json": self.kaist_calibrations,
            "camera.json": self.kaist_cameras,
            "lidar.json": self.kaist_lidars,
            "radar.json": self.kaist_radars,
            "category.json": self.kaist_categories,
            "dynamic_object.json": self.kaist_dynamic_objects,
            "ego_motion.json": self.kaist_ego_motions,
            "session_ego_motion.json": self.kaist_session_ego_motions,
            "hdmap.json": self.kaist_hdmaps,
        }
        
        # Also create empty placeholder files for tables we don't have data for
        placeholders = {
            "occupancy.json": [],
            "motion.json": [],
        }
        
        print(f"\nSaving KAIST-format files to: {self.output_root}")
        
        for filename, data in {**outputs, **placeholders}.items():
            filepath = self.output_root / filename
            with open(filepath, "w") as f:
                json.dump(self._to_dict_list(data) if data else [], f, indent=2)
            print(f"  Saved: {filename} ({len(data)} records)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate KAIST-format data from nuScenes mini dataset"
    )
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=Path("/user_data/nuscenes-mini"),
        help="Path to nuScenes mini dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/user_data/kaist-simulated"),
        help="Output directory for KAIST-format files",
    )
    
    args = parser.parse_args()
    
    # Allow override via environment
    nuscenes_root = Path(os.environ.get("NUSCENES_ROOT", args.nuscenes_root))
    output_root = Path(os.environ.get("KAIST_OUTPUT", args.output))
    
    converter = NuScenesToKAISTConverter(nuscenes_root, output_root)
    converter.convert()
    converter.save()
    
    print("\n" + "=" * 60)
    print("KAIST SIMULATION DATA GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nTo test ingestion, run:")
    print(f"  export KAIST_SOURCE_PATH={output_root}")
    print(f"  python -m kaist_ingestion.kaist_runner all")


if __name__ == "__main__":
    main()
