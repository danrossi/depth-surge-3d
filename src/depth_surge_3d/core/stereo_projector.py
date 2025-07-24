"""
Main StereoProjector class for 2D to 3D VR conversion.

This module provides the main orchestration class that coordinates all
processing steps using the modular utility functions.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from ..models.depth_estimator import DepthEstimator, create_depth_estimator
from ..utils.resolution import (
    get_resolution_dimensions, calculate_vr_output_dimensions,
    validate_resolution_settings, auto_detect_resolution
)
from ..utils.image_processing import (
    resize_image, normalize_depth_map, depth_to_disparity,
    create_shifted_image, apply_center_crop, apply_fisheye_distortion,
    apply_fisheye_square_crop, create_vr_frame, hole_fill_image
)
from ..utils.file_operations import (
    validate_video_file, get_video_properties, create_output_directories,
    get_frame_files, generate_output_filename
)
from ..utils.progress import ProgressTracker, create_progress_tracker
from ..processing.video_processor import VideoProcessor
from ..processing.batch_processor import BatchProcessor
from ..core.constants import DEFAULT_SETTINGS, VR_RESOLUTIONS


class StereoProjector:
    """
    Main class for converting 2D videos to 3D VR format.
    
    This class orchestrates the entire conversion process using modular
    utility functions and maintains minimal state.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize StereoProjector.
        
        Args:
            model_path: Path to depth estimation model
            device: Processing device ('auto', 'cuda', 'cpu')
        """
        self.depth_estimator = create_depth_estimator(model_path, device)
        self._model_loaded = False
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        vr_format: str = None,
        baseline: float = None,
        focal_length: float = None,
        keep_intermediates: bool = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        preserve_audio: bool = None,
        target_fps: Optional[int] = None,
        min_resolution: str = None,
        super_sample: str = None,
        apply_distortion: bool = None,
        fisheye_projection: str = None,
        fisheye_fov: float = None,
        crop_factor: float = None,
        vr_resolution: str = None,
        fisheye_crop_factor: float = None,
        hole_fill_quality: str = None,
        processing_mode: str = None
    ) -> bool:
        """
        Process video to create 3D VR version.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory path
            **kwargs: Processing parameters (defaults from DEFAULT_SETTINGS)
            
        Returns:
            True if processing completed successfully
        """
        # Apply defaults for None values
        settings = self._apply_default_settings(locals())
        
        try:
            # Validate inputs
            if not self._validate_inputs(video_path, output_dir, settings):
                return False
            
            # Ensure model is loaded
            if not self._ensure_model_loaded():
                return False
            
            # Get video properties
            video_props = get_video_properties(video_path)
            if not video_props:
                print(f"Error: Cannot read video properties from {video_path}")
                return False
            
            # Validate and resolve settings
            resolved_settings = self._resolve_settings(settings, video_props)
            
            # Create processor based on mode
            if resolved_settings['processing_mode'] == 'batch':
                processor = BatchProcessor(self.depth_estimator)
            else:
                processor = VideoProcessor(self.depth_estimator)
            
            # Process the video
            return processor.process(
                video_path=video_path,
                output_dir=output_dir,
                video_properties=video_props,
                settings=resolved_settings
            )
            
        except Exception as e:
            print(f"Error during video processing: {e}")
            return False
    
    def process_image(
        self,
        image_path: str,
        output_dir: str,
        **kwargs
    ) -> bool:
        """
        Process single image to create 3D stereo pair.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory path
            **kwargs: Processing parameters
            
        Returns:
            True if processing completed successfully
        """
        settings = self._apply_default_settings(kwargs)
        
        try:
            # Ensure model is loaded
            if not self._ensure_model_loaded():
                return False
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot load image from {image_path}")
                return False
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process image
            result = self._process_single_image(image, settings)
            if not result:
                return False
            
            left_img, right_img, vr_frame = result
            
            # Save results
            base_name = Path(image_path).stem
            cv2.imwrite(str(output_path / f"{base_name}_left.png"), left_img)
            cv2.imwrite(str(output_path / f"{base_name}_right.png"), right_img)
            cv2.imwrite(str(output_path / f"{base_name}_vr.png"), vr_frame)
            
            print(f"Image processing complete. Output saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error during image processing: {e}")
            return False
    
    def _apply_default_settings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default settings for None parameters."""
        settings = {}
        for key, default_value in DEFAULT_SETTINGS.items():
            # Get value from params, excluding 'self' and 'video_path', 'output_dir'
            if key in params and params[key] is not None:
                settings[key] = params[key]
            else:
                settings[key] = default_value
        
        # Handle special cases
        special_params = ['video_path', 'output_dir', 'start_time', 'end_time', 'target_fps', 'min_resolution']
        for param in special_params:
            if param in params:
                settings[param] = params[param]
        
        return settings
    
    def _validate_inputs(self, video_path: str, output_dir: str, settings: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        # Validate video file
        if not validate_video_file(video_path):
            print(f"Error: Invalid or unsupported video file: {video_path}")
            return False
        
        # Validate output directory
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}")
            return False
        
        return True
    
    def _ensure_model_loaded(self) -> bool:
        """Ensure the depth estimation model is loaded."""
        if not self._model_loaded:
            if self.depth_estimator.load_model():
                self._model_loaded = True
            else:
                return False
        return True
    
    def _resolve_settings(self, settings: Dict[str, Any], video_props: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve and validate settings based on video properties."""
        resolved = settings.copy()
        
        # Resolve VR resolution
        if resolved['vr_resolution'] == 'auto':
            resolved['vr_resolution'] = auto_detect_resolution(
                video_props['width'], 
                video_props['height'], 
                resolved['vr_format']
            )
        
        # Validate resolution settings
        validation = validate_resolution_settings(
            resolved['vr_resolution'],
            resolved['vr_format'],
            video_props['width'],
            video_props['height']
        )
        
        if not validation['valid']:
            print("Warning: Invalid resolution settings")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        for recommendation in validation['recommendations']:
            print(f"Recommendation: {recommendation}")
        
        # Get final resolution dimensions
        per_eye_width, per_eye_height = get_resolution_dimensions(resolved['vr_resolution'])
        vr_output_width, vr_output_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, resolved['vr_format']
        )
        
        resolved.update({
            'per_eye_width': per_eye_width,
            'per_eye_height': per_eye_height,
            'vr_output_width': vr_output_width,
            'vr_output_height': vr_output_height,
            'source_width': video_props['width'],
            'source_height': video_props['height'],
            'source_fps': video_props['fps']
        })
        
        return resolved
    
    def _process_single_image(
        self, 
        image: np.ndarray, 
        settings: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Process a single image to create stereo pair and VR frame.
        
        Args:
            image: Input image array
            settings: Processing settings
            
        Returns:
            Tuple of (left_image, right_image, vr_frame) or None if failed
        """
        try:
            # Get target dimensions
            per_eye_width = settings['per_eye_width']
            per_eye_height = settings['per_eye_height']
            
            # Resize image if needed
            current_height, current_width = image.shape[:2]
            if settings['super_sample'] != 'none':
                # Apply super sampling for better quality
                target_width = max(current_width, per_eye_width * 2)
                target_height = max(current_height, per_eye_height * 2)
                image = resize_image(image, target_width, target_height)
            
            # Generate depth map
            depth_map = self.depth_estimator.estimate_depth(image)
            depth_map = normalize_depth_map(depth_map)
            
            # Create stereo pair
            disparity_map = depth_to_disparity(
                depth_map, 
                settings['baseline'], 
                settings['focal_length']
            )
            
            left_img = create_shifted_image(image, disparity_map, "left")
            right_img = create_shifted_image(image, disparity_map, "right")
            
            # Apply hole filling
            if settings['hole_fill_quality'] in ['fast', 'advanced']:
                left_img = hole_fill_image(left_img, method=settings['hole_fill_quality'])
                right_img = hole_fill_image(right_img, method=settings['hole_fill_quality'])
            
            # Apply distortion if enabled
            if settings['apply_distortion']:
                left_img = apply_fisheye_distortion(
                    left_img, 
                    settings['fisheye_fov'], 
                    settings['fisheye_projection']
                )
                right_img = apply_fisheye_distortion(
                    right_img, 
                    settings['fisheye_fov'], 
                    settings['fisheye_projection']
                )
                
                # Apply fisheye-aware cropping
                left_final = apply_fisheye_square_crop(
                    left_img, per_eye_width, per_eye_height, settings['fisheye_crop_factor']
                )
                right_final = apply_fisheye_square_crop(
                    right_img, per_eye_width, per_eye_height, settings['fisheye_crop_factor']
                )
            else:
                # Apply center cropping
                left_cropped = apply_center_crop(left_img, settings['crop_factor'])
                right_cropped = apply_center_crop(right_img, settings['crop_factor'])
                
                # Resize to target dimensions
                left_final = resize_image(left_cropped, per_eye_width, per_eye_height)
                right_final = resize_image(right_cropped, per_eye_width, per_eye_height)
            
            # Create VR frame
            vr_frame = create_vr_frame(left_final, right_final, settings['vr_format'])
            
            return left_final, right_final, vr_frame
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.depth_estimator.get_model_info()
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.depth_estimator.unload_model()
        self._model_loaded = False


def create_stereo_projector(
    model_path: Optional[str] = None,
    device: str = 'auto'
) -> StereoProjector:
    """
    Factory function to create a StereoProjector instance.
    
    Args:
        model_path: Path to depth estimation model
        device: Processing device
        
    Returns:
        Configured StereoProjector instance
    """
    return StereoProjector(model_path, device) 