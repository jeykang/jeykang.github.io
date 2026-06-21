"""Stage 3: 2D object detection on camera frames using YOLOv11.

Runs a pretrained COCO model and filters to AD-relevant classes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# AD-relevant COCO class IDs and their names
AD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
}


@dataclass
class Detection2D:
    """A single 2D bounding box detection."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    camera_name: str


class Detector2D:
    """YOLOv11 2D object detector for autonomous driving."""

    def __init__(self, model_name: str = "yolo11x.pt", device: str = "cuda:0",
                 conf_threshold: float = 0.3):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        print(f"Loaded YOLO model: {model_name} on {device}")

    def detect_frame(
        self,
        image: np.ndarray,
        camera_name: str,
    ) -> List[Detection2D]:
        """Run detection on a single frame.

        Args:
            image: (H, W, 3) BGR image.
            camera_name: Source camera identifier.

        Returns:
            List of Detection2D for AD-relevant objects.
        """
        results = self.model(image, device=self.device, conf=self.conf_threshold,
                             verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in AD_CLASSES:
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                detections.append(Detection2D(
                    class_id=cls_id,
                    class_name=AD_CLASSES[cls_id],
                    confidence=conf,
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    camera_name=camera_name,
                ))
        return detections

    def detect_clip(
        self,
        camera_frames: Dict[str, list],
        cameras: Optional[List[str]] = None,
    ) -> Dict[int, List[Detection2D]]:
        """Run detection on all frames of a clip, keyed by LiDAR spin index.

        Args:
            camera_frames: Dict[sensor_name -> List[CameraFrame]] from decode.
            cameras: Which cameras to process. None = all.

        Returns:
            Dict[spin_index -> List[Detection2D]] across all cameras.
        """
        sensors = cameras or list(camera_frames.keys())
        detections_by_spin = {}

        for sensor in sensors:
            frames = camera_frames.get(sensor, [])
            for idx, frame in enumerate(frames):
                dets = self.detect_frame(frame.image, sensor)
                if idx not in detections_by_spin:
                    detections_by_spin[idx] = []
                detections_by_spin[idx].extend(dets)

        return detections_by_spin
