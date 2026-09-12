"""
YOLO Object Detection module for ForeSite.
Provides:
1. Ultralytics YOLO detector (PyTorch backend, when installed)
2. ONNX Runtime YOLO detector (ultra-lightweight, zero-PyTorch dependency)
3. Simulated fallback detector for deterministic testing and offline demo reliability
4. Label normalization to canonical ForeSite classes: worker, forklift, excavator, truck, loader
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import os
import logging
import cv2
import numpy as np

from .config import TrackingConfig

logger = logging.getLogger(__name__)

# Standard COCO 80 classes mapping
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class BaseDetector(ABC):
    """Abstract detector interface returning plain Python structures."""

    def __init__(self, config: TrackingConfig):
        self.config = config

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single frame.

        Returns:
            List of detections:
            [
                {
                    "class_name": "worker",
                    "bbox": [x1, y1, x2, y2],  # pixel coordinates
                    "confidence": 0.94
                },
                ...
            ]
        """
        pass

    def normalize_label(self, raw_label: str) -> Optional[str]:
        """
        Map detector label to canonical ForeSite label.
        Returns None if label is irrelevant to construction safety.
        """
        raw_clean = raw_label.lower().strip()
        canonical = self.config.class_mapping.get(raw_clean, raw_clean)

        if canonical in self.config.target_classes:
            return canonical
        return None


class ONNXYOLODetector(BaseDetector):
    """
    Lightweight YOLOv8 ONNX inference engine using onnxruntime or cv2.dnn.
    Requires ~20MB runtime footprint, ideal for constrained environments.
    """

    def __init__(self, model_path: str, config: TrackingConfig):
        super().__init__(config)
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None

        try:
            import onnxruntime as ort
            # Use CPU or CoreML execution provider if available on Mac
            providers = ["CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CoreMLExecutionProvider")
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"Initialized ONNX YOLO detector with {model_path} on {providers[0]}")
        except Exception as e:
            logger.warning(f"Could not initialize onnxruntime: {e}. Falling back to OpenCV DNN.")
            self.net = cv2.dnn.readNetFromONNX(model_path)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize to 640x640 for YOLOv8."""
        orig_h, orig_w = frame.shape[:2]
        target_size = 640
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to 640x640 with grey (114, 114, 114)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # HWC -> CHW, BGR -> RGB, normalize to [0, 1]
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, (pad_x, pad_y)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        blob, scale, (pad_x, pad_y) = self._preprocess(frame)
        orig_h, orig_w = frame.shape[:2]

        if self.session is not None:
            outputs = self.session.run(self.output_names, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            outputs = self.net.forward()

        # Output shape: (1, 84, 8400) -> transpose to (8400, 84)
        predictions = np.squeeze(outputs).T
        boxes = []
        confidences = []
        class_ids = []

        for row in predictions:
            classes_scores = row[4:]
            max_score = float(np.max(classes_scores))
            if max_score >= self.config.conf_threshold:
                class_id = int(np.argmax(classes_scores))
                raw_label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                canonical = self.normalize_label(raw_label)
                if canonical is None:
                    continue

                # cx, cy, w, h in 640x640 coordinate space
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                # Shift back from padding and unscale to original frame dimensions
                x1 = (cx - w / 2.0 - pad_x) / scale
                y1 = (cy - h / 2.0 - pad_y) / scale
                x2 = (cx + w / 2.0 - pad_x) / scale
                y2 = (cy + h / 2.0 - pad_y) / scale

                # Clamp to frame dimensions
                x1 = max(0.0, min(float(orig_w), x1))
                y1 = max(0.0, min(float(orig_h), y1))
                x2 = max(0.0, min(float(orig_w), x2))
                y2 = max(0.0, min(float(orig_h), y2))

                boxes.append([x1, y1, x2 - x1, y2 - y1])  # for cv2.dnn.NMSBoxes
                confidences.append(max_score)
                class_ids.append(class_id)

        detections = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.config.conf_threshold, self.config.iou_threshold)
            for idx in indices:
                i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                x, y, w, h = boxes[i]
                raw_label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else "unknown"
                canonical_label = self.normalize_label(raw_label)
                detections.append({
                    "class_name": canonical_label,
                    "bbox": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                    "confidence": round(float(confidences[i]), 4),
                })

        return detections


class UltralyticsYOLODetector(BaseDetector):
    """Detector using Ultralytics YOLO (PyTorch backend)."""

    def __init__(self, model_name_or_path: str, config: TrackingConfig):
        super().__init__(config)
        from ultralytics import YOLO
        self.model = YOLO(model_name_or_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_idx = int(box.cls[0])
            raw_label = res.names.get(cls_idx, f"class_{cls_idx}")
            canonical = self.normalize_label(raw_label)

            if canonical is None:
                continue

            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())

            detections.append({
                "class_name": canonical,
                "bbox": [round(float(c), 2) for c in xyxy],
                "confidence": round(conf, 4),
            })

        return detections


class SyntheticFallbackDetector(BaseDetector):
    """
    Deterministic simulated detector for testing, mock pipelines, and backup demos.
    Generates a converging worker and forklift scenario.
    Worker crosses from left to right; Forklift approaches from right to left.
    """

    def __init__(self, config: TrackingConfig):
        super().__init__(config)
        self.step = 0

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        self.step += 1

        # Progress fraction through sequence (smooth monotonic progress over 120 frames)
        t = min(1.0, max(0.0, (self.step - 1) / 120.0))

        # Worker walking from left (x=200, y=500) towards center (x=620, y=390)
        # Scaled dynamically with frame width and height relative to base 1280x720
        sx = w / 1280.0
        sy = h / 720.0

        wx = (200.0 + 420.0 * t) * sx
        wy = (500.0 - 110.0 * t) * sy
        w_x1 = wx - 20.0 * sx
        w_y1 = wy - 90.0 * sy
        w_x2 = wx + 20.0 * sx
        w_y2 = wy

        # Forklift approaching from right (x=1050, y=350) towards center (x=680, y=410)
        fx = (1050.0 - 370.0 * t) * sx
        fy = (350.0 + 60.0 * t) * sy
        f_x1 = fx - 72.0 * sx
        f_y1 = fy - 124.0 * sy
        f_x2 = fx + 50.0 * sx
        f_y2 = fy + 8.0 * sy

        return [
            {
                "class_name": "worker",
                "bbox": [
                    round(w_x1, 2),
                    round(w_y1, 2),
                    round(w_x2, 2),
                    round(w_y2, 2),
                ],
                "confidence": 0.94,
            },
            {
                "class_name": "forklift",
                "bbox": [
                    round(f_x1, 2),
                    round(f_y1, 2),
                    round(f_x2, 2),
                    round(f_y2, 2),
                ],
                "confidence": 0.89,
            }
        ]


def create_detector(config: Optional[TrackingConfig] = None, force_synthetic: bool = False) -> BaseDetector:
    """
    Factory creating the appropriate detector instance based on available dependencies and weights.
    Falls back gracefully to ONNX or Synthetic to guarantee 100% demo uptime.
    """
    cfg = config or TrackingConfig()

    if force_synthetic:
        logger.info("Using SyntheticFallbackDetector as requested.")
        return SyntheticFallbackDetector(cfg)

    # 1. Try Ultralytics YOLO (PyTorch backend, auto-downloads weights if needed)
    try:
        from ultralytics import YOLO
        logger.info(f"Loading Ultralytics YOLO detector with model: {cfg.model_name_or_path}")
        return UltralyticsYOLODetector(cfg.model_name_or_path, cfg)
    except ImportError:
        logger.info("Ultralytics not installed; checking ONNX / Synthetic fallbacks.")
    except Exception as e:
        logger.warning(f"Ultralytics YOLO initialization failed: {e}")

    # 2. Try ONNX model if specified or found
    onnx_candidate = cfg.model_name_or_path.replace(".pt", ".onnx")
    if os.path.exists(onnx_candidate):
        logger.info(f"Loading ONNX YOLO model from {onnx_candidate}")
        return ONNXYOLODetector(onnx_candidate, cfg)

    # 3. Fallback to synthetic detector for testing and offline demo execution
    logger.info("Using SyntheticFallbackDetector (deterministic demo & test mode).")
    return SyntheticFallbackDetector(cfg)
