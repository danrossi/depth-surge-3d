"""Unit tests for motion compensator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.depth_surge_3d.models.motion_compensator import MotionCompensator


class TestMotionCompensator:
    """Test motion compensator functionality."""

    def test_initialization(self):
        """Test motion compensator initialization."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)
        assert compensator.flow_estimator == mock_flow_estimator

    def test_warp_depth_with_flow(self):
        """Test depth map warping with optical flow."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        depth = np.random.rand(480, 640).astype(np.float32)
        flow = np.random.randn(480, 640, 2).astype(np.float32) * 5

        warped = compensator.warp_depth_with_flow(depth, flow)

        assert warped.shape == depth.shape
        assert warped.dtype == np.float32
        assert 0 <= warped.min() <= 1.0
        assert 0 <= warped.max() <= 1.0

    def test_blend_depths_basic(self):
        """Test basic depth blending."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        original = np.full((480, 640), 0.5, dtype=np.float32)
        compensated = np.full((480, 640), 0.8, dtype=np.float32)

        # 50% blend
        blended = compensator.blend_depths(original, compensated, alpha=0.5)

        assert blended.shape == original.shape
        assert np.allclose(blended, 0.65)  # (0.5 * 0.5 + 0.5 * 0.8)

    def test_blend_depths_alpha_zero(self):
        """Test blending with alpha=0 returns original."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        original = np.random.rand(480, 640).astype(np.float32)
        compensated = np.random.rand(480, 640).astype(np.float32)

        blended = compensator.blend_depths(original, compensated, alpha=0.0)

        assert np.allclose(blended, original)

    def test_blend_depths_alpha_one(self):
        """Test blending with alpha=1 returns compensated."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        original = np.random.rand(480, 640).astype(np.float32)
        compensated = np.random.rand(480, 640).astype(np.float32)

        blended = compensator.blend_depths(original, compensated, alpha=1.0)

        assert np.allclose(blended, compensated)

    def test_blend_depths_with_occlusion_mask(self):
        """Test blending with occlusion mask."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        original = np.full((10, 10), 0.5, dtype=np.float32)
        compensated = np.full((10, 10), 0.8, dtype=np.float32)
        occlusion_mask = np.zeros((10, 10), dtype=bool)
        occlusion_mask[5:, 5:] = True  # Mark bottom-right as occluded

        blended = compensator.blend_depths(
            original, compensated, alpha=0.5, occlusion_mask=occlusion_mask
        )

        # Non-occluded region should be blended
        assert np.allclose(blended[:5, :5], 0.65)
        # Occluded region should use original
        assert np.allclose(blended[5:, 5:], 0.5)

    def test_detect_scene_cuts_no_cuts(self):
        """Test scene cut detection with no cuts."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        # Create frames with small gradual changes
        frames = np.array([np.full((480, 640, 3), i * 5, dtype=np.uint8) for i in range(10)])

        cuts = compensator.detect_scene_cuts(frames, threshold=30.0)

        assert len(cuts) == 0

    def test_detect_scene_cuts_with_cuts(self):
        """Test scene cut detection with scene changes."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        # Create frames with hard cuts
        frames = []
        for i in range(3):
            frames.append(np.full((480, 640, 3), 50, dtype=np.uint8))
        # Hard cut at index 3
        for i in range(3):
            frames.append(np.full((480, 640, 3), 200, dtype=np.uint8))

        frames = np.array(frames)

        cuts = compensator.detect_scene_cuts(frames, threshold=30.0)

        assert len(cuts) > 0
        assert 3 in cuts  # Should detect cut at frame 3

    def test_detect_scene_cuts_single_frame(self):
        """Test scene cut detection with single frame."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        frames = np.array([np.zeros((480, 640, 3), dtype=np.uint8)])

        cuts = compensator.detect_scene_cuts(frames)

        assert cuts == []

    def test_detect_occlusions(self):
        """Test occlusion detection via forward-backward consistency."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        # Create consistent flow (forward and backward cancel out)
        forward_flow = np.ones((480, 640, 2), dtype=np.float32) * 5
        backward_flow = -forward_flow  # Perfect consistency

        occlusion_mask = compensator.detect_occlusions(forward_flow, backward_flow, threshold=1.0)

        assert occlusion_mask.shape == (480, 640)
        assert occlusion_mask.dtype == bool
        # Should have minimal occlusions with consistent flow
        assert np.sum(occlusion_mask) < 0.1 * occlusion_mask.size

    def test_compensate_depth_batch_no_frames(self):
        """Test depth compensation with insufficient frames."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        frames = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
        depth_maps = np.random.rand(1, 480, 640).astype(np.float32)

        compensated = compensator.compensate_depth_batch(frames, depth_maps)

        # Should return copy of original
        assert compensated.shape == depth_maps.shape
        assert np.allclose(compensated, depth_maps)

    def test_compensate_depth_batch_mismatched_shapes(self):
        """Test depth compensation with mismatched frame/depth counts."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        frames = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
        depth_maps = np.random.rand(3, 480, 640).astype(np.float32)

        with pytest.raises(ValueError, match="Number of frames must match"):
            compensator.compensate_depth_batch(frames, depth_maps)

    def test_compensate_depth_batch_basic(self):
        """Test basic depth compensation."""
        mock_flow_estimator = MagicMock()
        # Mock flow estimation to return small flows
        mock_flow_estimator.estimate_flow_batch.return_value = np.zeros(
            (9, 480, 640, 2), dtype=np.float32
        )

        compensator = MotionCompensator(mock_flow_estimator)

        frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        depth_maps = np.random.rand(10, 480, 640).astype(np.float32)

        compensated = compensator.compensate_depth_batch(frames, depth_maps, blend_alpha=0.5)

        assert compensated.shape == depth_maps.shape
        assert compensated.dtype == np.float32
        # First frame should be unchanged
        assert np.allclose(compensated[0], depth_maps[0])

    def test_compensate_depth_batch_with_scene_cuts(self):
        """Test depth compensation skips frames after scene cuts."""
        mock_flow_estimator = MagicMock()
        mock_flow_estimator.estimate_flow_batch.return_value = np.zeros(
            (5, 480, 640, 2), dtype=np.float32
        )

        compensator = MotionCompensator(mock_flow_estimator)

        # Create frames with scene cut at index 3
        frames = []
        for i in range(3):
            frames.append(np.full((480, 640, 3), 50, dtype=np.uint8))
        for i in range(3):
            frames.append(np.full((480, 640, 3), 200, dtype=np.uint8))
        frames = np.array(frames)

        depth_maps = np.random.rand(6, 480, 640).astype(np.float32)

        compensated = compensator.compensate_depth_batch(
            frames, depth_maps, blend_alpha=0.5, scene_cut_threshold=30.0
        )

        assert compensated.shape == depth_maps.shape
        # Frame after cut should use original depth
        assert np.allclose(compensated[3], depth_maps[3])

    def test_measure_temporal_consistency(self):
        """Test temporal consistency measurement."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        # Create depth maps with known variance
        depth_maps = np.array(
            [
                np.full((10, 10), 0.5, dtype=np.float32),
                np.full((10, 10), 0.6, dtype=np.float32),
                np.full((10, 10), 0.7, dtype=np.float32),
            ]
        )

        consistency = compensator.measure_temporal_consistency(depth_maps)

        # Variance should be 0.1 (constant difference between frames)
        assert pytest.approx(consistency, abs=0.01) == 0.1

    def test_measure_temporal_consistency_single_frame(self):
        """Test temporal consistency with single frame."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        depth_maps = np.random.rand(1, 480, 640).astype(np.float32)

        consistency = compensator.measure_temporal_consistency(depth_maps)

        assert consistency == 0.0

    def test_get_compensation_stats(self):
        """Test compensation statistics calculation."""
        mock_flow_estimator = MagicMock()
        compensator = MotionCompensator(mock_flow_estimator)

        # Original with high variance
        original = np.array(
            [
                np.full((10, 10), 0.0, dtype=np.float32),
                np.full((10, 10), 1.0, dtype=np.float32),
                np.full((10, 10), 0.0, dtype=np.float32),
            ]
        )

        # Compensated with lower variance
        compensated = np.array(
            [
                np.full((10, 10), 0.4, dtype=np.float32),
                np.full((10, 10), 0.6, dtype=np.float32),
                np.full((10, 10), 0.4, dtype=np.float32),
            ]
        )

        stats = compensator.get_compensation_stats(original, compensated)

        assert "original_temporal_variance" in stats
        assert "compensated_temporal_variance" in stats
        assert "improvement_percentage" in stats
        assert "mean_absolute_change" in stats

        # Compensated should have lower variance
        assert stats["compensated_temporal_variance"] < stats["original_temporal_variance"]
        assert stats["improvement_percentage"] > 0
