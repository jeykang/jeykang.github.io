"""
FUSE Mount Manager for Nvidia PhysicalAI zip archives.

Mounts STORED-mode zip archives via ratarmount so that the Parquet files
inside appear as regular files.  This enables zero-copy Iceberg
registration — no data is extracted or duplicated.

Uses ratarmount (pip install ratarmount) which can mount an entire
directory of zip archives as a single FUSE filesystem.  This means one
mount per sensor directory (~30 mounts total for the full dataset)
instead of one mount per zip file (45,000+).

Runs on the HOST (not inside the Spark container).  The mount root is
volume-mounted into the Spark container via docker-compose.yml so that
Spark/Iceberg can read the exposed Parquet files at query time.

Prerequisites:
    pip install ratarmount

Usage:
    # Mount all sensors
    python -m nvidia_ingestion.fuse_mount mount --all

    # Mount a specific sensor group
    python -m nvidia_ingestion.fuse_mount mount --sensor radar
    python -m nvidia_ingestion.fuse_mount mount --sensor lidar

    # Unmount everything
    python -m nvidia_ingestion.fuse_mount unmount --all

    # Show status
    python -m nvidia_ingestion.fuse_mount status
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


DEFAULT_SOURCE = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)
DEFAULT_FUSE_ROOT = "/tmp/nvidia-fuse"


def _ensure_ratarmount() -> str:
    """Return the path to ratarmount, or raise if not installed."""
    path = shutil.which("ratarmount")
    if not path:
        raise RuntimeError(
            "ratarmount is not installed.  Install with: pip install ratarmount"
        )
    return path


def _discover_sensor_dirs(source: str) -> Dict[str, List[str]]:
    """Discover all sensor directories that contain zip archives."""
    sensors = {}

    # Lidar
    lidar_dir = os.path.join(source, "lidar", "lidar_top_360fov")
    if os.path.isdir(lidar_dir):
        sensors["lidar/lidar_top_360fov"] = [lidar_dir]

    # Radar
    radar_dir = os.path.join(source, "radar")
    if os.path.isdir(radar_dir):
        for sub in sorted(os.listdir(radar_dir)):
            full = os.path.join(radar_dir, sub)
            if os.path.isdir(full) and glob.glob(os.path.join(full, "*.zip")):
                sensors[f"radar/{sub}"] = [full]

    # Camera
    cam_dir = os.path.join(source, "camera")
    if os.path.isdir(cam_dir):
        for sub in sorted(os.listdir(cam_dir)):
            full = os.path.join(cam_dir, sub)
            if os.path.isdir(full) and glob.glob(os.path.join(full, "*.zip")):
                sensors[f"camera/{sub}"] = [full]

    # Labels (egomotion)
    ego_dir = os.path.join(source, "labels", "egomotion")
    if os.path.isdir(ego_dir):
        sensors["labels/egomotion"] = [ego_dir]

    return sensors


def mount_sensor(
    source_dir: str,
    mount_point: str,
    ratarmount_bin: str,
    recursive: bool = True,
) -> bool:
    """Mount a sensor directory's zip archives via ratarmount.

    All zips in source_dir are exposed as a single FUSE filesystem at
    mount_point.
    """
    if os.path.ismount(mount_point):
        return True  # already mounted

    os.makedirs(mount_point, exist_ok=True)

    cmd = [ratarmount_bin]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([source_dir, mount_point])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for large dirs
        )
        if result.returncode != 0:
            print(f"  [FAIL] {source_dir} → {mount_point}")
            if result.stderr:
                print(f"         {result.stderr.strip()[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {source_dir} — ratarmount took >10min")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def unmount_point(mount_point: str) -> bool:
    """Unmount a single FUSE mount."""
    if not os.path.ismount(mount_point):
        try:
            os.rmdir(mount_point)
        except OSError:
            pass
        return True

    for unmount_cmd in ["fusermount3 -u", "fusermount -u", "umount"]:
        try:
            subprocess.run(
                unmount_cmd.split() + [mount_point],
                check=True, capture_output=True, text=True,
            )
            try:
                os.rmdir(mount_point)
            except OSError:
                pass
            return True
        except subprocess.CalledProcessError:
            continue

    print(f"  [FAIL] could not unmount {mount_point}")
    return False


def get_active_mounts(fuse_root: str = DEFAULT_FUSE_ROOT) -> List[str]:
    """Return list of active FUSE mount points under the root."""
    active = []
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith(fuse_root):
                    active.append(parts[1])
    except OSError:
        pass
    return sorted(active)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_mount(args):
    """Mount zip archives as FUSE directories via ratarmount."""
    ratarmount_bin = _ensure_ratarmount()
    source = args.source
    fuse_root = args.fuse_root

    all_sensors = _discover_sensor_dirs(source)
    if not all_sensors:
        print("No sensor directories with zip files found.")
        return

    if args.sensor:
        # Filter to matching sensors
        filtered = {k: v for k, v in all_sensors.items()
                    if k.startswith(args.sensor)}
        if not filtered:
            print(f"No sensor directories matching '{args.sensor}'")
            print(f"Available: {', '.join(sorted(all_sensors.keys()))}")
            return
        all_sensors = filtered
    elif not args.all:
        print("Specify --sensor <name> or --all")
        sys.exit(1)

    print(f"Mounting {len(all_sensors)} sensor directory/ies via ratarmount...")
    ok, fail = 0, 0
    for sensor_path, source_dirs in sorted(all_sensors.items()):
        mount_point = os.path.join(fuse_root, sensor_path)
        print(f"  {sensor_path} → {mount_point}")
        for src in source_dirs:
            if mount_sensor(src, mount_point, ratarmount_bin):
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} mounted, {fail} failed")
    if ok > 0:
        print(f"Parquet files accessible under {fuse_root}/")


def cmd_unmount(args):
    """Unmount all FUSE mounts under the root."""
    mounts = get_active_mounts(args.fuse_root)
    if not mounts:
        print("No active FUSE mounts found.")
        return

    print(f"Unmounting {len(mounts)} mount(s)...")
    for mp in reversed(mounts):
        if unmount_point(mp):
            print(f"  [OK] {mp}")
        else:
            print(f"  [FAIL] {mp}")


def cmd_status(args):
    """Show active FUSE mounts and file counts."""
    mounts = get_active_mounts(args.fuse_root)
    if not mounts:
        print("No active FUSE mounts.")
        return

    print(f"Active FUSE mounts under {args.fuse_root}:\n")
    total_files = 0
    for mp in mounts:
        try:
            parquets = glob.glob(os.path.join(mp, "**/*.parquet"), recursive=True)
            total_files += len(parquets)
            print(f"  {mp}  ({len(parquets)} parquet files)")
        except Exception:
            print(f"  {mp}  (could not list)")

    print(f"\nTotal: {len(mounts)} mounts, {total_files} Parquet files")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FUSE mount manager for Nvidia PhysicalAI zip archives"
    )
    sub = parser.add_subparsers(dest="command")

    # mount
    p_mount = sub.add_parser("mount", help="Mount zip archives via ratarmount")
    p_mount.add_argument("--source", default=DEFAULT_SOURCE,
                         help="Dataset root path")
    p_mount.add_argument("--fuse-root", default=DEFAULT_FUSE_ROOT,
                         help="FUSE mount root directory")
    p_mount.add_argument("--sensor", type=str, default=None,
                         help="Sensor prefix: lidar, radar, camera, labels")
    p_mount.add_argument("--all", action="store_true",
                         help="Mount all sensor groups")
    p_mount.set_defaults(func=cmd_mount)

    # unmount
    p_unmount = sub.add_parser("unmount", help="Unmount all FUSE mounts")
    p_unmount.add_argument("--fuse-root", default=DEFAULT_FUSE_ROOT)
    p_unmount.add_argument("--all", action="store_true")
    p_unmount.set_defaults(func=cmd_unmount)

    # status
    p_status = sub.add_parser("status", help="Show active mount status")
    p_status.add_argument("--fuse-root", default=DEFAULT_FUSE_ROOT)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
