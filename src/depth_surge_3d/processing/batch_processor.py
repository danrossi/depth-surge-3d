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
    
    Note: This is currently a placeholder that extends VideoProcessor.
    Full batch processing implementation would include parallel processing
    of operations grouped by type (all depth maps, then all stereo pairs, etc.)
    """
    
    def __init__(self, depth_estimator: DepthEstimator):
        super().__init__(depth_estimator)
    
    def process(
        self,
        video_path: str,
        output_dir: str,
        video_properties: Dict[str, Any],
        settings: Dict[str, Any]
    ) -> bool:
        """
        Process video in batch mode.
        
        Currently delegates to serial processing.
        Future implementation will include true batch processing with:
        - Parallel frame extraction
        - Batch depth map generation
        - Parallel stereo pair creation
        - Batch fisheye distortion
        - Parallel VR frame creation
        
        Args:
            video_path: Path to input video
            output_dir: Output directory path
            video_properties: Video metadata
            settings: Processing settings
            
        Returns:
            True if processing completed successfully
        """
        print("Note: Batch processing currently uses serial implementation")
        print("Future versions will include true parallel batch processing")
        
        # Temporarily delegate to serial processing
        # TODO: Implement true batch processing with parallel operations
        return super().process(video_path, output_dir, video_properties, settings) 