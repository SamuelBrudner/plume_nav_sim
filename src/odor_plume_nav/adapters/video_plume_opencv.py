"""
VideoPlume module providing a simple implementation for video-based odor plume environments.
"""

import numpy as np
from pathlib import Path
import cv2
from contextlib import suppress
from typing import Dict, Union, Optional

from odor_plume_nav.video_plume_config import VideoPlumeConfig
from odor_plume_nav.config.utils import load_config


class VideoPlume:
    """Minimal VideoPlume for testing and factory purposes."""
    
    @classmethod
    def from_config(
        cls,
        video_path: Union[str, Path],
        config_dict: Optional[Dict] = None,
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> 'VideoPlume':
        """
        Create a VideoPlume from configuration.
        """
        params: Dict = {}
        if config_path is not None:
            cfg = load_config(config_path)
            params |= cfg.get("video_plume", {})
        if config_dict is not None:
            params |= config_dict
        params["video_path"] = video_path
        params |= kwargs
        try:
            validated = VideoPlumeConfig.model_validate(params)
            return cls(**validated.model_dump())
        except Exception as e:
            raise ValueError(f"Invalid VideoPlume configuration: {e}") from e

    def __init__(
        self,
        video_path,
        flip: bool = False,
        kernel_size: int = 0,
        kernel_sigma: float = 1.0,
        **kwargs
    ):
        """
        Initialize with basic parameters.
        """
        self.video_path = Path(video_path)
        self.flip = flip
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self._is_closed = False
        self.cap = None

        if not self.video_path.exists():
            raise IOError(f"Video file does not exist: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))

    def get_frame(self, frame_idx: int):
        """Get a grayscale frame by index, or None if invalid."""
        if self._is_closed:
            raise ValueError("Cannot get frame from closed VideoPlume")
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret and self.flip:
            frame = cv2.flip(frame, 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None

    @property
    def duration(self) -> float:
        return 0.0 if self.fps == 0 else self.frame_count / self.fps

    @property
    def shape(self) -> tuple:
        return (self.height, self.width)

    def get_metadata(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "shape": self.shape
        }

    def close(self) -> None:
        if not self._is_closed and self.cap is not None:
            self.cap.release()
            self._is_closed = True

    def __del__(self):
        with suppress(Exception):
            self.close()
