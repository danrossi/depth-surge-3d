"""
Processing pipeline orchestrator.

Coordinates the complete video processing pipeline across all specialized processors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...io.operations import (
    create_output_directories,
    save_processing_settings,
    update_processing_status,
)
from ...utils.path_utils import generate_output_filename
from ...utils import (
    step_complete,
    saved_to,
    title_bar,
    success as console_success,
)


class ProcessingOrchestrator:
    """
    High-level pipeline control and step sequencing.

    Responsibilities:
    - Pipeline execution flow
    - Step sequencing
    - Progress tracking coordination
    - Error handling
    - Settings management
    - Console output formatting
    """

    def __init__(
        self,
        depth_processor,
        stereo_generator,
        distortion_processor,
        upscaler,
        vr_assembler,
        video_encoder,
        verbose: bool = False,
    ):
        """
        Initialize processing orchestrator.

        Args:
            depth_processor: DepthMapProcessor instance
            stereo_generator: StereoPairGenerator instance
            distortion_processor: DistortionProcessor instance
            upscaler: FrameUpscalerProcessor instance
            vr_assembler: VRFrameAssembler instance
            video_encoder: VideoEncoder instance
            verbose: Enable verbose output
        """
        self.depth_processor = depth_processor
        self.stereo_generator = stereo_generator
        self.distortion_processor = distortion_processor
        self.upscaler = upscaler
        self.vr_assembler = vr_assembler
        self.video_encoder = video_encoder
        self.verbose = verbose
        self._settings_file = None  # Track settings file for error handling
        self._total_steps = 7  # Updated dynamically based on settings

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Main processing pipeline entry point.

        Args:
            video_path: Input video path
            output_dir: Output directory
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Console output
            - Filesystem state management
            - Delegates to all processor modules
        """
        try:
            # Setup processing environment
            output_path, directories, self._settings_file = self._setup_processing(
                str(video_path), str(output_dir), settings, {}
            )

            # Execute processing pipeline
            success = self._execute_pipeline(
                str(video_path),
                output_path,
                directories,
                settings,
                progress_tracker,
            )

            return success

        except Exception as e:
            print(f"Error in video processing: {e}")
            if self._settings_file:
                update_processing_status(self._settings_file, "failed", {"error": str(e)})
            return False

    def _execute_pipeline(
        self,
        video_path: str,
        output_path: Path,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Execute complete 8-step pipeline.

        Args:
            video_path: Input video path
            output_path: Output directory path
            directories: Dictionary of processing directories
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if all steps successful, False otherwise

        Side effects:
            - Executes all pipeline steps
            - Progress updates
            - Console output
        """
        # Calculate total steps based on settings
        self._total_steps = self._get_total_steps(settings)

        # Step 1: Extract frames (delegated to depth_processor)
        frame_files, fps = self.depth_processor.extract_frames(
            video_path, directories, settings, progress_tracker
        )
        if not frame_files:
            return False

        # Step 2: Generate depth maps (delegated to depth_processor)
        depth_maps = self.depth_processor.generate_depth_maps(
            frame_files, settings, directories, progress_tracker
        )
        if depth_maps is None:
            return False

        # Execute steps 3-8
        return self._execute_remaining_steps(
            directories,
            settings,
            frame_files,
            fps,
            video_path,
            output_path,
            progress_tracker,
        )

    def _execute_remaining_steps(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        frame_files: list[Path],
        fps: float,
        video_path: str,
        output_path: Path,
        progress_tracker=None,
        current_step: int = 3,
    ) -> bool:
        """
        Execute remaining pipeline steps after frame extraction.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings
            frame_files: List of extracted frame files
            fps: Video frames per second
            video_path: Input video path
            output_path: Output directory path
            progress_tracker: Optional progress tracker
            current_step: Current step number

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes steps 2-8 of pipeline
            - Progress updates
        """
        num_frames = len(frame_files)

        # Step 3: Create stereo pairs (delegated to stereo_generator)
        if not self.stereo_generator.create_stereo_pairs(
            frame_files, directories, settings, progress_tracker, current_step, self._total_steps
        ):
            return self._handle_step_error("Stereo pair creation failed")
        current_step += 1

        # Step 4: Apply fisheye distortion (optional - delegated to distortion_processor)
        if settings.get("apply_distortion", True):
            if not self.distortion_processor.apply_distortion(
                directories, settings, progress_tracker, current_step, self._total_steps
            ):
                return self._handle_step_error("Distortion failed")
            current_step += 1

        # Step 5: Crop frames (delegated to vr_assembler)
        if not self.vr_assembler.crop_frames(
            directories, settings, progress_tracker, current_step, self._total_steps, num_frames
        ):
            return self._handle_step_error("Frame cropping failed")
        current_step += 1

        # Step 6: Apply AI upscaling (optional - delegated to upscaler)
        if settings.get("upscale_model", "none") != "none":
            if not self.upscaler.apply_upscaling(
                directories, settings, progress_tracker, current_step, self._total_steps
            ):
                return self._handle_step_error("Upscaling failed")
            current_step += 1

        # Step 7: Assemble VR frames (delegated to vr_assembler)
        if not self.vr_assembler.assemble_vr_frames(
            directories, settings, progress_tracker, current_step, self._total_steps, num_frames
        ):
            return self._handle_step_error("VR frame assembly failed")
        current_step += 1

        # Step 8: Create final video (delegated to video_encoder)
        success = self.video_encoder.create_output_video(
            directories,
            output_path,
            video_path,
            settings,
            progress_tracker,
            current_step,
            self._total_steps,
        )

        # Finalize and cleanup
        if progress_tracker and hasattr(progress_tracker, "finish"):
            progress_tracker.finish("Video processing complete")
        self._finalize_processing(success, output_path, video_path, settings, num_frames)
        return success

    def _setup_processing(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
        video_properties: dict[str, Any],
    ) -> tuple[Path, dict[str, Path], Path | None]:
        """
        Setup processing directories and settings file.

        Args:
            video_path: Input video path
            output_dir: Output directory
            settings: Processing settings
            video_properties: Video metadata

        Returns:
            Tuple of (output_path, directories, settings_file)

        Side effects:
            - Creates directories
            - Writes settings file
            - Console output
        """
        output_path = Path(output_dir)
        directories = create_output_directories(output_path, settings["keep_intermediates"])
        batch_name = f"{Path(video_path).stem}_{int(time.time())}"
        settings_file = save_processing_settings(
            output_path, batch_name, settings, video_properties, video_path
        )

        print(f"\n{title_bar('=== Depth Surge 3D Video Processing ===')}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print("Using Video-Depth-Anything for temporal consistency\n")

        return output_path, directories, settings_file

    def _finalize_processing(
        self,
        success: bool,
        output_path: Path,
        video_path: str,
        settings: dict[str, Any],
        num_frames: int,
    ) -> None:
        """
        Finalize processing and update settings file.

        Args:
            success: Whether processing succeeded
            output_path: Output directory path
            video_path: Input video path
            settings: Processing settings
            num_frames: Number of frames processed

        Side effects:
            - Updates settings file with completion status
            - Console output
        """
        if success:
            print(console_success("Processing complete!"))
            if self._settings_file:
                output_filename = generate_output_filename(
                    Path(video_path).name,
                    settings["vr_format"],
                    settings["vr_resolution"],
                )
                update_processing_status(
                    self._settings_file,
                    "completed",
                    {
                        "final_output": str(output_path / output_filename),
                        "frames_processed": num_frames,
                    },
                )
        elif self._settings_file:
            update_processing_status(
                self._settings_file, "failed", {"error": "Video creation failed"}
            )

    @staticmethod
    def _get_total_steps(settings: dict[str, Any]) -> int:
        """
        PURE: Calculate total steps based on settings.

        Args:
            settings: Processing settings

        Returns:
            Total number of pipeline steps (6-8)
        """
        total = 6  # Base: Extract, Depth, Stereo, Crop, Assemble, Video
        if settings.get("apply_distortion", True):
            total += 1  # Add Step 4: Distortion
        if settings.get("upscale_model", "none") != "none":
            total += 1  # Add Step 6: Upscaling
        return total  # 6-8 steps total

    def _update_step_progress(
        self,
        progress_tracker,
        message: str,
        step_name: str,
        progress: int,
        total: int,
    ) -> None:
        """
        Update progress for a processing step.

        Args:
            progress_tracker: Progress tracker instance
            message: Progress message
            step_name: Name of current step
            progress: Current progress value
            total: Total progress value

        Side effects:
            - Updates progress tracker state
        """
        if progress_tracker:
            progress_tracker.update_progress(
                message,
                phase="processing",
                frame_num=progress,
                step_name=step_name,
                step_progress=progress,
                step_total=total,
            )

    def _print_step_complete(
        self, num_items: int, duration: float, item_type: str = "frames"
    ) -> None:
        """
        Print step completion message.

        Args:
            num_items: Number of items processed
            duration: Processing duration in seconds
            item_type: Type of items processed

        Side effects:
            - Console output
        """
        print(step_complete(f"Processed {num_items:04d} {item_type} in {duration:.2f}s"))

    def _print_saved_to(self, directory: Path, message_prefix: str = "Saved to") -> None:
        """
        Print save location message.

        Args:
            directory: Directory path
            message_prefix: Message prefix text

        Side effects:
            - Console output
        """
        if directory:
            print(saved_to(f"{message_prefix}: {directory}\n"))
        else:
            print()

    def _handle_step_error(self, error_msg: str) -> bool:
        """
        Handle step failure and update settings file.

        Args:
            error_msg: Error message

        Returns:
            False (always returns False to indicate failure)

        Side effects:
            - Console output
            - Updates settings file with error status
        """
        print(f"Error: {error_msg}")
        if self._settings_file:
            update_processing_status(self._settings_file, "failed", {"error": error_msg})
        return False
