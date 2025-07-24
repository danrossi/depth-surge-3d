"""
Batch processor for parallel processing mode.

This module handles batch processing where operations are grouped by type
and can utilize parallel processing for better performance.
"""

from typing import Dict, Any

from ..models.depth_estimator import DepthEstimator
from .video_processor import VideoProcessor


class BatchProcessor(VideoProcessor):
    """
    Handles batch video processing with parallel operations.
    """
    def __init__(self, depth_estimator: DepthEstimator, verbose: bool = False):
        super().__init__(depth_estimator, verbose=verbose)

    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: Dict[str, Any],
        settings: Dict[str, Any],
        progress_callback=None
    ) -> bool:
        """
        Process video in batch mode.
        Currently delegates to serial processing.
        """
        print("\n=== Depth Surge 3D Batch Processing Pipeline ===")
        print(f"Input: {video_path}")
        print(f"Output: {output_dir}")
        print(f"Mode: batch\n")
        # Temporarily delegate to serial processing (stepwise CLI output)
        return super().process(video_path, output_dir, video_properties, settings, progress_callback=progress_callback) 