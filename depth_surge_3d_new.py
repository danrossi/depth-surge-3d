#!/usr/bin/env python3
"""
Depth Surge 3D - Convert 2D videos to immersive 3D VR format using AI depth estimation.

This is the main entry point using the new modular architecture.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.depth_surge_3d.core.stereo_projector import create_stereo_projector
from src.depth_surge_3d.core.constants import (
    DEFAULT_SETTINGS, VR_RESOLUTIONS, FISHEYE_PROJECTIONS, 
    HOLE_FILL_METHODS, VALIDATION_RANGES
)
from src.depth_surge_3d.utils.resolution import get_available_resolutions
from src.depth_surge_3d.utils.file_operations import validate_video_file


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert 2D videos to immersive 3D VR format using AI depth estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with auto-detection
  python depth_surge_3d_new.py input_video.mp4
  
  # High quality 4K conversion
  python depth_surge_3d_new.py input_video.mp4 --vr-resolution 16x9-4k --format over_under
  
  # Custom resolution for ultra-wide content  
  python depth_surge_3d_new.py input_video.mp4 --vr-resolution custom:2560x1080 --format over_under
  
  # Fast batch processing
  python depth_surge_3d_new.py input_video.mp4 --processing-mode batch --vr-resolution 16x9-720p
  
  # Process specific time segment
  python depth_surge_3d_new.py input_video.mp4 --start 01:30 --end 03:45
        """
    )
    
    # Positional arguments
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('output_dir', nargs='?', default=DEFAULT_SETTINGS['output_dir'] if 'output_dir' in DEFAULT_SETTINGS else './output',
                       help='Output directory (default: ./output)')
    
    # VR format and resolution
    parser.add_argument('-f', '--format', choices=['side_by_side', 'over_under'], 
                       default=DEFAULT_SETTINGS['vr_format'], 
                       help=f'VR output format (default: {DEFAULT_SETTINGS["vr_format"]})')
    
    parser.add_argument('--vr-resolution', 
                       default=DEFAULT_SETTINGS['vr_resolution'], 
                       help='VR output resolution format. Preset options: ' + 
                            ', '.join(list(VR_RESOLUTIONS.keys())[:10]) + '... ' +
                            'Custom format: custom:WIDTHxHEIGHT (e.g., custom:1920x1080). ' +
                            f'Default: {DEFAULT_SETTINGS["vr_resolution"]}')
    
    # Processing settings
    parser.add_argument('--processing-mode', choices=['serial', 'batch'], 
                       default=DEFAULT_SETTINGS['processing_mode'],
                       help=f'Processing mode - serial (frame-by-frame) or batch (task-by-task + parallel) (default: {DEFAULT_SETTINGS["processing_mode"]})')
    
    # Time range
    parser.add_argument('-s', '--start', help='Start time in mm:ss or hh:mm:ss format (e.g., 01:30)')
    parser.add_argument('-e', '--end', help='End time in mm:ss or hh:mm:ss format (e.g., 03:45)')
    
    # Depth and stereo parameters
    parser.add_argument('-b', '--baseline', type=float, default=DEFAULT_SETTINGS['baseline'],
                       help=f'Stereo baseline distance in meters (default: {DEFAULT_SETTINGS["baseline"]} - average human IPD)')
    parser.add_argument('--focal-length', type=int, default=DEFAULT_SETTINGS['focal_length'],
                       help=f'Camera focal length in pixels (default: {DEFAULT_SETTINGS["focal_length"]})')
    
    # Distortion and projection
    parser.add_argument('--fisheye-projection', choices=FISHEYE_PROJECTIONS,
                       default=DEFAULT_SETTINGS['fisheye_projection'],
                       help=f'Fisheye projection type (default: {DEFAULT_SETTINGS["fisheye_projection"]})')
    parser.add_argument('--fisheye-fov', type=float, default=DEFAULT_SETTINGS['fisheye_fov'],
                       help=f'Fisheye field of view in degrees (default: {DEFAULT_SETTINGS["fisheye_fov"]})')
    parser.add_argument('--no-distortion', action='store_true',
                       help='Disable fisheye distortion (keeps rectilinear projection)')
    
    # Quality and processing options
    parser.add_argument('--crop-factor', type=float, default=DEFAULT_SETTINGS['crop_factor'],
                       help=f'Center crop factor (1.0 = no crop, 0.5 = crop to half) (default: {DEFAULT_SETTINGS["crop_factor"]})')
    parser.add_argument('--fisheye-crop-factor', type=float, default=DEFAULT_SETTINGS['fisheye_crop_factor'],
                       help=f'Fisheye crop factor (default: {DEFAULT_SETTINGS["fisheye_crop_factor"]})')
    parser.add_argument('--hole-fill-quality', choices=HOLE_FILL_METHODS,
                       default=DEFAULT_SETTINGS['hole_fill_quality'],
                       help=f'Hole filling quality (default: {DEFAULT_SETTINGS["hole_fill_quality"]})')
    
    # Model and device
    parser.add_argument('--model', help='Path to Depth-Anything-V2 model file (auto-downloads if missing)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Processing device (default: auto)')
    
    # Output options
    parser.add_argument('--no-audio', action='store_true',
                       help='Do not preserve audio in output')
    parser.add_argument('--no-intermediates', action='store_true',
                       help='Do not keep intermediate processing files')
    parser.add_argument('--target-fps', type=int,
                       help='Target output FPS (default: match source)')
    
    # Information and debugging
    parser.add_argument('--list-resolutions', action='store_true',
                       help='List all available VR resolution options')
    parser.add_argument('--model-info', action='store_true',
                       help='Show model information and exit')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser


def validate_arguments(args) -> bool:
    """Validate command line arguments."""
    
    # Validate input video
    if not validate_video_file(args.input_video):
        print(f"Error: Invalid or unsupported video file: {args.input_video}")
        return False
    
    # Validate ranges
    if args.baseline < VALIDATION_RANGES['baseline'][0] or args.baseline > VALIDATION_RANGES['baseline'][1]:
        print(f"Error: Baseline must be between {VALIDATION_RANGES['baseline'][0]} and {VALIDATION_RANGES['baseline'][1]} meters")
        return False
    
    if args.focal_length < VALIDATION_RANGES['focal_length'][0] or args.focal_length > VALIDATION_RANGES['focal_length'][1]:
        print(f"Error: Focal length must be between {VALIDATION_RANGES['focal_length'][0]} and {VALIDATION_RANGES['focal_length'][1]} pixels")
        return False
    
    if args.fisheye_fov < VALIDATION_RANGES['fisheye_fov'][0] or args.fisheye_fov > VALIDATION_RANGES['fisheye_fov'][1]:
        print(f"Error: FOV must be between {VALIDATION_RANGES['fisheye_fov'][0]} and {VALIDATION_RANGES['fisheye_fov'][1]} degrees")
        return False
    
    if args.crop_factor < VALIDATION_RANGES['crop_factor'][0] or args.crop_factor > VALIDATION_RANGES['crop_factor'][1]:
        print(f"Error: Crop factor must be between {VALIDATION_RANGES['crop_factor'][0]} and {VALIDATION_RANGES['crop_factor'][1]}")
        return False
    
    if args.target_fps and (args.target_fps < VALIDATION_RANGES['target_fps'][0] or args.target_fps > VALIDATION_RANGES['target_fps'][1]):
        print(f"Error: Target FPS must be between {VALIDATION_RANGES['target_fps'][0]} and {VALIDATION_RANGES['target_fps'][1]}")
        return False
    
    return True


def list_available_resolutions():
    """List all available VR resolution options."""
    print("Available VR Resolution Options:")
    print("=" * 40)
    
    resolutions = get_available_resolutions()
    
    for category, items in resolutions.items():
        if items:  # Only show categories with items
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                print(f"  {item['name']:<15} - {item['description']}")
    
    print(f"\nCustom Resolution:")
    print(f"  custom:WxH      - Custom resolution (e.g., custom:1920x1080)")
    print(f"\nAuto Detection:")
    print(f"  auto            - Automatically detect optimal resolution")


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_resolutions:
        list_available_resolutions()
        return 0
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Create stereo projector
    try:
        projector = create_stereo_projector(args.model, args.device)
        
        if args.model_info:
            info = projector.get_model_info()
            print("Model Information:")
            print("=" * 20)
            for key, value in info.items():
                print(f"{key}: {value}")
            return 0
        
        # Process video
        print(f"Starting Depth Surge 3D processing...")
        print(f"Input: {args.input_video}")
        print(f"Output: {args.output_dir}")
        print(f"Mode: {args.processing_mode}")
        print(f"Format: {args.format}")
        print(f"Resolution: {args.vr_resolution}")
        
        success = projector.process_video(
            video_path=args.input_video,
            output_dir=args.output_dir,
            vr_format=args.format,
            vr_resolution=args.vr_resolution,
            processing_mode=args.processing_mode,
            baseline=args.baseline,
            focal_length=args.focal_length,
            start_time=args.start,
            end_time=args.end,
            apply_distortion=not args.no_distortion,
            fisheye_projection=args.fisheye_projection,
            fisheye_fov=args.fisheye_fov,
            crop_factor=args.crop_factor,
            fisheye_crop_factor=args.fisheye_crop_factor,
            hole_fill_quality=args.hole_fill_quality,
            preserve_audio=not args.no_audio,
            keep_intermediates=not args.no_intermediates,
            target_fps=args.target_fps
        )
        
        if success:
            print("🎉 Processing completed successfully!")
            return 0
        else:
            print("❌ Processing failed. Check error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Clean up
        try:
            if 'projector' in locals():
                projector.unload_model()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main()) 