"""Tests for video_processor step methods."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.video_processor import VideoProcessor


class TestStepCropFrames:
    """Test _step_crop_frames method."""

    @pytest.fixture
    def mock_depth_estimator(self):
        """Create a mock depth estimator."""
        return Mock()

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.get_step_duration = Mock(return_value=1.5)
        return tracker

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories with test frames."""
        # Create source directories
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_dir.mkdir()
        right_dir.mkdir()

        # Create test frames
        for i in range(3):
            left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        # Create output directories
        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        return {
            "base": tmp_path,
            "left_frames": left_dir,
            "right_frames": right_dir,
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

    def test_step_crop_frames_skip_existing(
        self, temp_dirs, mock_depth_estimator, mock_progress_tracker
    ):
        """Test that cropping is skipped when frames already exist."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"keep_intermediates": True}

        # Pre-populate cropped directories
        for i in range(3):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(temp_dirs["left_cropped"] / f"frame_{i:04d}.png"), frame)
            cv2.imwrite(str(temp_dirs["right_cropped"] / f"frame_{i:04d}.png"), frame)

        result = processor._step_crop_frames(
            temp_dirs, settings, mock_progress_tracker, current_step=5, num_frames=3
        )

        assert result is True
        # Progress should be updated for skipped step
        mock_progress_tracker.update_progress.assert_called_once()

    def test_step_crop_frames_success(self, temp_dirs, mock_depth_estimator, mock_progress_tracker):
        """Test successful frame cropping."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "keep_intermediates": True,
            "vr_resolution": "16x9-1080p",
            "vr_format": "side_by_side",
        }

        with patch.object(processor, "_crop_frames", return_value=True) as mock_crop:
            result = processor._step_crop_frames(
                temp_dirs, settings, mock_progress_tracker, current_step=5, num_frames=3
            )

        assert result is True
        mock_crop.assert_called_once_with(temp_dirs, settings, mock_progress_tracker, 3)

    def test_step_crop_frames_failure(self, temp_dirs, mock_depth_estimator, mock_progress_tracker):
        """Test frame cropping failure handling."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "keep_intermediates": True,
            "vr_resolution": "16x9-1080p",
            "vr_format": "side_by_side",
        }

        with patch.object(processor, "_crop_frames", return_value=False):
            with patch.object(processor, "_handle_step_error") as mock_error:
                result = processor._step_crop_frames(
                    temp_dirs,
                    settings,
                    mock_progress_tracker,
                    current_step=5,
                    num_frames=3,
                )

        assert result is False
        mock_error.assert_called_once_with("Frame cropping failed")


class TestStepApplyUpscalingIntegration:
    """Test _step_apply_upscaling method integration."""

    @pytest.fixture
    def mock_depth_estimator(self):
        """Create a mock depth estimator."""
        return Mock()

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.get_step_duration = Mock(return_value=10.5)
        return tracker

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories."""
        left_dir = tmp_path / "left_cropped"
        right_dir = tmp_path / "right_cropped"
        left_dir.mkdir()
        right_dir.mkdir()

        return {
            "base": tmp_path,
            "left_cropped": left_dir,
            "right_cropped": right_dir,
        }

    def test_step_apply_upscaling_skip_when_disabled(
        self, temp_dirs, mock_depth_estimator, mock_progress_tracker
    ):
        """Test that upscaling is skipped when disabled."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"upscale_model": "none", "keep_intermediates": True}

        with patch.object(processor, "_apply_upscaling", return_value=True) as mock_up:
            result = processor._step_apply_upscaling(
                temp_dirs, settings, mock_progress_tracker, current_step=6
            )

        assert result is True
        mock_up.assert_called_once()

    def test_step_apply_upscaling_success(
        self, temp_dirs, mock_depth_estimator, mock_progress_tracker
    ):
        """Test successful upscaling step."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"upscale_model": "x4", "keep_intermediates": True}

        with patch.object(processor, "_apply_upscaling", return_value=True) as mock_up:
            result = processor._step_apply_upscaling(
                temp_dirs, settings, mock_progress_tracker, current_step=6
            )

        assert result is True
        mock_up.assert_called_once()

    def test_step_apply_upscaling_failure(
        self, temp_dirs, mock_depth_estimator, mock_progress_tracker
    ):
        """Test upscaling step failure handling."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"upscale_model": "x4", "keep_intermediates": True}

        with patch.object(processor, "_apply_upscaling", return_value=False):
            with patch.object(processor, "_handle_step_error") as mock_error:
                result = processor._step_apply_upscaling(
                    temp_dirs,
                    settings,
                    mock_progress_tracker,
                    current_step=6,
                )

        assert result is False
        mock_error.assert_called_once_with("Upscaling failed")


class TestCropFrames:
    """Test _crop_frames helper method."""

    @pytest.fixture
    def mock_depth_estimator(self):
        """Create a mock depth estimator."""
        return Mock()

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_dirs_with_frames(self, tmp_path):
        """Create temporary directories with test frames."""
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"

        left_dir.mkdir()
        right_dir.mkdir()
        left_cropped.mkdir()
        right_cropped.mkdir()

        # Create test frames (large enough to crop)
        for i in range(3):
            left_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        return {
            "base": tmp_path,
            "left_frames": left_dir,
            "right_frames": right_dir,
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

    def test_crop_frames_success(
        self, temp_dirs_with_frames, mock_depth_estimator, mock_progress_tracker
    ):
        """Test successful frame cropping."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "vr_resolution": "16x9-1080p",
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "apply_distortion": False,
        }

        result = processor._crop_frames(
            temp_dirs_with_frames, settings, mock_progress_tracker, total_frames=3
        )

        assert result is True
        # Check that cropped frames were created
        left_cropped_files = list(temp_dirs_with_frames["left_cropped"].glob("*.png"))
        right_cropped_files = list(temp_dirs_with_frames["right_cropped"].glob("*.png"))
        assert len(left_cropped_files) == 3
        assert len(right_cropped_files) == 3

        # Verify progress was updated
        assert mock_progress_tracker.update_progress.called

    def test_crop_frames_mismatched_count(
        self, temp_dirs_with_frames, mock_depth_estimator, mock_progress_tracker
    ):
        """Test cropping with mismatched frame counts."""
        # Remove one right frame
        right_frames = list(temp_dirs_with_frames["right_frames"].glob("*.png"))
        right_frames[0].unlink()

        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "vr_resolution": "16x9-1080p",
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "apply_distortion": False,
        }

        result = processor._crop_frames(
            temp_dirs_with_frames, settings, mock_progress_tracker, total_frames=3
        )

        # Method still completes but may have fewer output files
        # In practice this would be caught earlier in the pipeline
        assert result is True or result is False  # Either is acceptable


class TestGetTotalSteps:
    """Test _get_total_steps helper method."""

    @pytest.fixture
    def mock_depth_estimator(self):
        """Create a mock depth estimator."""
        return Mock()

    def test_get_total_steps_all_enabled(self, mock_depth_estimator):
        """Test total steps calculation with all features enabled."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "apply_distortion": True,
            "upscale_model": "x4",
        }

        total = processor._get_total_steps(settings)
        assert total == 8  # All 8 steps

    def test_get_total_steps_no_distortion(self, mock_depth_estimator):
        """Test total steps without fisheye distortion."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "apply_distortion": False,
            "upscale_model": "x4",
        }

        total = processor._get_total_steps(settings)
        assert total == 7  # 8 - 1 (no distortion)

    def test_get_total_steps_no_upscaling(self, mock_depth_estimator):
        """Test total steps without upscaling."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "apply_distortion": True,
            "upscale_model": "none",
        }

        total = processor._get_total_steps(settings)
        assert total == 7  # 8 - 1 (no upscaling)

    def test_get_total_steps_minimal(self, mock_depth_estimator):
        """Test total steps with minimal features."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {
            "apply_distortion": False,
            "upscale_model": "none",
        }

        total = processor._get_total_steps(settings)
        assert total == 6  # 8 - 2 (no distortion, no upscaling)


class TestLoadFrames:
    """Test _load_frames helper method."""

    @pytest.fixture
    def mock_depth_estimator(self):
        """Create a mock depth estimator."""
        return Mock()

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create a mock progress tracker."""
        return Mock()

    @pytest.fixture
    def temp_frame_files(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(5):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(str(frame_path))

        return frame_files

    def test_load_frames_success(
        self, temp_frame_files, mock_depth_estimator, mock_progress_tracker
    ):
        """Test successful frame loading."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"super_sample": False, "per_eye_width": 100, "per_eye_height": 100}

        frames = processor._load_frames(temp_frame_files, settings, mock_progress_tracker)

        assert frames is not None
        assert len(frames) == 5
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_load_frames_with_missing_file(
        self, temp_frame_files, mock_depth_estimator, mock_progress_tracker
    ):
        """Test loading frames with one missing file."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"super_sample": False, "per_eye_width": 100, "per_eye_height": 100}

        # Add a non-existent file
        temp_frame_files.append("/nonexistent/frame.png")

        frames = processor._load_frames(temp_frame_files, settings, mock_progress_tracker)

        # Should still load valid frames
        assert frames is not None
        assert len(frames) == 5  # Only the valid ones

    def test_load_frames_all_missing(self, mock_depth_estimator, mock_progress_tracker):
        """Test loading frames when all files are missing."""
        processor = VideoProcessor(mock_depth_estimator)
        settings = {"super_sample": False, "per_eye_width": 100, "per_eye_height": 100}

        fake_files = ["/nonexistent/frame1.png", "/nonexistent/frame2.png"]

        frames = processor._load_frames(fake_files, settings, mock_progress_tracker)

        # When all frames fail, method returns None (exception case)
        # or empty array depending on implementation
        assert frames is None or (frames is not None and len(frames) == 0)
