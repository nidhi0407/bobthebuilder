"""
Position extraction and coordinate normalization utilities for ForeSite.
Converts bounding boxes into representative ground/image coordinates and normalizes
them to [0.0, 1.0] so downstream trajectory prediction (Person 2) is invariant to video resolution.
"""

from typing import List, Tuple, Union


def extract_representative_position(
    bbox: List[Union[float, int]],
    anchor: str = "bottom-center"
) -> Tuple[float, float]:
    """
    Extract a single representative (x, y) point from a bounding box [x1, y1, x2, y2].

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        anchor: 'bottom-center' (default) or 'center'.
                'bottom-center' approximates feet/ground contact for workers and wheels/tracks for machinery.

    Returns:
        (x, y) pixel coordinates.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of 4 elements [x1, y1, x2, y2], got {bbox}")

    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    if anchor == "bottom-center":
        x = (x1 + x2) / 2.0
        y = y2
    elif anchor == "center":
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
    else:
        raise ValueError(f"Unknown anchor: '{anchor}'. Supported anchors: 'bottom-center', 'center'")

    return x, y


def normalize_coordinates(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
    clamp: bool = True
) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to [0.0, 1.0] range relative to frame dimensions.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.
        clamp: If True, restricts coordinates to [0.0, 1.0].

    Returns:
        (x_norm, y_norm) floats in range [0.0, 1.0].
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Frame dimensions must be positive, got {frame_width}x{frame_height}")

    x_norm = float(x) / float(frame_width)
    y_norm = float(y) / float(frame_height)

    if clamp:
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

    return round(x_norm, 5), round(y_norm, 5)


def denormalize_coordinates(
    x_norm: float,
    y_norm: float,
    frame_width: int,
    frame_height: int
) -> Tuple[int, int]:
    """
    Convert normalized [0.0, 1.0] coordinates back to integer pixel coordinates.
    Useful for visualization and bounding box mapping in Person 4 dashboard.

    Args:
        x_norm: Normalized horizontal coordinate.
        y_norm: Normalized vertical coordinate.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        (x_pixel, y_pixel) integers.
    """
    x_pix = int(round(x_norm * frame_width))
    y_pix = int(round(y_norm * frame_height))
    return x_pix, y_pix
