"""
Ball Detection Module - TrackNet Implementation

State-of-the-art tennis ball detection using TrackNet deep learning architecture.

Usage:
    from ball_detection import TrackNet, BallDetector
    from ball_detection.tracknet_dataset import TrackNetDataset
"""

from .tracknet_model import TrackNet, TrackNetLightning, create_tracknet
from .detect_ball import BallDetector

__all__ = [
    'TrackNet',
    'TrackNetLightning',
    'create_tracknet',
    'BallDetector'
]

__version__ = '1.0.0'
