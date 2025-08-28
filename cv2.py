import numpy as np

# Minimal constants used in tests and code
CAP_PROP_FRAME_WIDTH = 0
CAP_PROP_FRAME_HEIGHT = 1
CAP_PROP_FPS = 5
CAP_PROP_FRAME_COUNT = 7
CAP_PROP_POS_FRAMES = 1

COLOR_BGR2GRAY = 6


def VideoCapture(*args, **kwargs):
    return _StubVideoCapture()


class _StubVideoCapture:
    def __init__(self):
        self._opened = True
        self._frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def isOpened(self):
        return self._opened

    def get(self, prop):
        return 0

    def read(self):
        return True, self._frame.copy()

    def set(self, prop, value):
        return True

    def release(self):
        self._opened = False


def flip(frame, code):
    # Basic flip implementation; code -1 flips vertically and horizontally
    if code == -1:
        return np.flipud(np.fliplr(frame))
    elif code == 0:
        return np.flipud(frame)
    elif code == 1:
        return np.fliplr(frame)
    return frame


def cvtColor(frame, code):
    if code == COLOR_BGR2GRAY and frame.ndim == 3:
        return frame.mean(axis=2).astype(frame.dtype)
    return frame


def GaussianBlur(frame, ksize, sigma):
    # Simple stub: return frame unchanged
    return frame
