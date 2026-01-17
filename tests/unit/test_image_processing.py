"""Unit tests for image processing utilities."""

import numpy as np
import pytest
from src.depth_surge_3d.utils.image_processing import (
    normalize_depth_map,
    depth_to_disparity,
)


class TestNormalizeDepthMap:
    """Test depth map normalization function."""

    def test_normalize_standard_range(self):
        """Test normalization of standard depth map."""
        depth = np.array([[10.0, 20.0], [30.0, 40.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == depth.shape

    def test_normalize_already_0_to_1(self):
        """Test normalization of already normalized depth map."""
        depth = np.array([[0.0, 0.5], [0.75, 1.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_constant_depth(self):
        """Test normalization of constant depth map returns zeros."""
        depth = np.full((100, 100), 5.0)
        normalized = normalize_depth_map(depth)

        # Constant depth should return all zeros
        assert np.all(normalized == 0.0)

    def test_normalize_negative_values(self):
        """Test normalization handles negative values."""
        depth = np.array([[-10.0, -5.0], [0.0, 5.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_large_values(self):
        """Test normalization with large depth values."""
        depth = np.array([[1000.0, 2000.0], [3000.0, 4000.0]])
        normalized = normalize_depth_map(depth)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_single_pixel(self):
        """Test normalization of single pixel."""
        depth = np.array([[42.0]])
        normalized = normalize_depth_map(depth)

        # Single value should become 0 (since min == max)
        assert normalized[0, 0] == 0.0

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves shape."""
        shapes = [(100, 100), (50, 200), (1920, 1080)]

        for shape in shapes:
            depth = np.random.rand(*shape) * 100
            normalized = normalize_depth_map(depth)
            assert normalized.shape == shape

    def test_normalize_3d_array(self):
        """Test normalization with 3D array."""
        depth = np.random.rand(10, 10, 3) * 100
        normalized = normalize_depth_map(depth)

        assert normalized.shape == depth.shape
        # Each channel should be normalized independently
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0


class TestDepthToDisparity:
    """Test depth to disparity conversion function."""

    def test_depth_to_disparity_basic(self):
        """Test basic depth to disparity conversion."""
        depth_map = np.array([[0.5, 0.75], [0.25, 1.0]])
        baseline = 0.1  # 10cm
        focal_length = 1000.0  # pixels

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        # Disparity should be inversely proportional to depth
        # Closer objects (lower depth values) should have higher disparity
        assert disparity.shape == depth_map.shape
        assert np.all(disparity > 0)

        # Check inverse relationship: lower depth = higher disparity
        assert disparity[1, 0] > disparity[0, 0]  # 0.25 depth > 0.5 depth disparity

    def test_depth_to_disparity_zero_depth_handling(self):
        """Test that zero depth values are handled safely."""
        depth_map = np.array([[0.0, 0.5], [1.0, 0.0]])
        baseline = 0.1
        focal_length = 1000.0

        # Should not raise division by zero error
        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        assert not np.isnan(disparity).any()
        assert not np.isinf(disparity).any()

    def test_depth_to_disparity_very_small_depth(self):
        """Test handling of very small depth values."""
        depth_map = np.array([[0.0001, 0.0005], [0.001, 0.01]])
        baseline = 0.1
        focal_length = 1000.0

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        # Very small depths should produce high disparity values
        assert np.all(disparity > 0)
        assert not np.isnan(disparity).any()

    def test_depth_to_disparity_uniform_depth(self):
        """Test disparity conversion with uniform depth."""
        depth_map = np.full((50, 50), 0.5)
        baseline = 0.1
        focal_length = 1000.0

        disparity = depth_to_disparity(depth_map, baseline, focal_length)

        # All values should be equal for uniform depth
        assert np.allclose(disparity, disparity[0, 0])

    def test_depth_to_disparity_baseline_scaling(self):
        """Test that disparity scales with baseline."""
        depth_map = np.array([[0.5, 0.75]])
        focal_length = 1000.0

        disparity_1 = depth_to_disparity(depth_map, baseline=0.1, focal_length=focal_length)
        disparity_2 = depth_to_disparity(depth_map, baseline=0.2, focal_length=focal_length)

        # Doubling baseline should double disparity
        assert np.allclose(disparity_2, disparity_1 * 2)

    def test_depth_to_disparity_focal_length_scaling(self):
        """Test that disparity scales with focal length."""
        depth_map = np.array([[0.5, 0.75]])
        baseline = 0.1

        disparity_1 = depth_to_disparity(depth_map, baseline, focal_length=1000.0)
        disparity_2 = depth_to_disparity(depth_map, baseline, focal_length=2000.0)

        # Doubling focal length should double disparity
        assert np.allclose(disparity_2, disparity_1 * 2)

    def test_depth_to_disparity_preserves_shape(self):
        """Test that conversion preserves array shape."""
        shapes = [(100, 100), (50, 200), (1920, 1080)]

        for shape in shapes:
            depth_map = np.random.rand(*shape)
            disparity = depth_to_disparity(depth_map, baseline=0.1, focal_length=1000.0)
            assert disparity.shape == shape

    def test_depth_to_disparity_zero_baseline(self):
        """Test with zero baseline (edge case)."""
        depth_map = np.array([[0.5, 1.0]])
        disparity = depth_to_disparity(depth_map, baseline=0.0, focal_length=1000.0)

        # Zero baseline should produce zero disparity
        assert np.allclose(disparity, 0.0)

    def test_depth_to_disparity_zero_focal_length(self):
        """Test with zero focal length (edge case)."""
        depth_map = np.array([[0.5, 1.0]])
        disparity = depth_to_disparity(depth_map, baseline=0.1, focal_length=0.0)

        # Zero focal length should produce zero disparity
        assert np.allclose(disparity, 0.0)
