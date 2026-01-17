"""Unit tests for optical flow estimator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.depth_surge_3d.models.optical_flow_estimator import (
    RAFTFlowEstimator,
    UniMatchFlowEstimator,
    create_optical_flow_estimator,
)


class TestRAFTFlowEstimator:
    """Test RAFT optical flow estimator."""

    def test_initialization(self):
        """Test RAFT estimator initialization."""
        estimator = RAFTFlowEstimator(device="cpu", model_size="small")
        assert estimator.device == "cpu"
        assert estimator.model_size == "small"
        assert estimator.model_type == "raft_small"
        assert estimator.model is None

    def test_load_model_success(self):
        """Test successful RAFT model loading."""
        with patch("torchvision.models.optical_flow.raft_small") as mock_raft:
            mock_model = MagicMock()
            # Configure mock so .to() returns self for chaining
            mock_model.to.return_value = mock_model
            mock_raft.return_value = mock_model

            estimator = RAFTFlowEstimator(device="cpu", model_size="small")
            result = estimator.load_model()

            assert result is True
            assert estimator.model is not None
            mock_model.to.assert_called_once_with("cpu")
            mock_model.eval.assert_called_once()

    def test_load_model_import_error(self):
        """Test RAFT model loading with import error."""
        with patch(
            "torchvision.models.optical_flow.raft_small",
            side_effect=ImportError("torchvision not installed"),
        ):
            estimator = RAFTFlowEstimator(device="cpu", model_size="small")
            result = estimator.load_model()

            assert result is False
            assert estimator.model is None

    def test_estimate_flow_shape(self):
        """Test optical flow estimation output shape."""
        estimator = RAFTFlowEstimator(device="cpu", model_size="small")

        # Mock loaded model
        mock_model = MagicMock()
        mock_flow = torch.randn(1, 2, 480, 640)  # [B, 2, H, W]
        mock_model.return_value = [mock_flow] * 12  # RAFT returns list of 12 flows
        estimator.model = mock_model

        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        flow = estimator.estimate_flow(frame1, frame2)

        assert flow.shape == (480, 640, 2)
        assert flow.dtype == np.float32

    def test_estimate_flow_batch(self):
        """Test batch optical flow estimation."""
        estimator = RAFTFlowEstimator(device="cpu", model_size="small")

        # Mock loaded model
        mock_model = MagicMock()
        mock_flow = torch.randn(1, 2, 480, 640)
        mock_model.return_value = [mock_flow] * 12
        estimator.model = mock_model

        frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)

        flows = estimator.estimate_flow_batch(frames)

        assert flows.shape == (9, 480, 640, 2)  # N-1 flows for N frames
        assert flows.dtype == np.float32

    def test_estimate_flow_without_loaded_model(self):
        """Test flow estimation without loaded model raises error."""
        estimator = RAFTFlowEstimator(device="cpu")

        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            estimator.estimate_flow(frame1, frame2)

    def test_estimate_flow_batch_insufficient_frames(self):
        """Test batch estimation with insufficient frames raises error."""
        estimator = RAFTFlowEstimator(device="cpu")
        estimator.model = MagicMock()  # Mock model

        frames = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Need at least 2 frames"):
            estimator.estimate_flow_batch(frames)

    def test_unload_model(self):
        """Test model unloading."""
        estimator = RAFTFlowEstimator(device="cpu")
        estimator.model = MagicMock()

        estimator.unload_model()

        assert estimator.model is None

    def test_get_model_info(self):
        """Test getting model information."""
        estimator = RAFTFlowEstimator(device="cpu", model_size="large")
        estimator.model = MagicMock()

        info = estimator.get_model_info()

        assert info["model_type"] == "raft_large"
        assert info["device"] == "cpu"
        assert info["loaded"] is True


class TestUniMatchFlowEstimator:
    """Test UniMatch optical flow estimator."""

    def test_initialization(self):
        """Test UniMatch estimator initialization."""
        estimator = UniMatchFlowEstimator(device="cpu")
        assert estimator.device == "cpu"
        assert estimator.model_type == "unimatch"
        assert estimator.model is None

    def test_load_model_not_installed(self):
        """Test UniMatch model loading when package not installed."""
        # UniMatch import happens inside try/except, just test the fallback behavior
        estimator = UniMatchFlowEstimator(device="cpu")
        result = estimator.load_model()

        # Should return False when UniMatch not installed (which it likely isn't)
        assert result is False

    def test_run_inference_not_implemented(self):
        """Test UniMatch inference raises NotImplementedError."""
        estimator = UniMatchFlowEstimator(device="cpu")
        estimator.model = MagicMock()  # Mock loaded model

        with pytest.raises(NotImplementedError):
            estimator._run_inference(torch.randn(1, 3, 480, 640), torch.randn(1, 3, 480, 640))


class TestFactoryFunction:
    """Test optical flow estimator factory function."""

    def test_create_raft_estimator_auto(self):
        """Test factory creates RAFT when auto mode and UniMatch unavailable."""
        with patch(
            "src.depth_surge_3d.models.optical_flow_estimator._try_load_unimatch",
            return_value=None,
        ):
            with patch(
                "src.depth_surge_3d.models.optical_flow_estimator._try_load_raft_large"
            ) as mock_raft:
                mock_estimator = MagicMock()
                mock_raft.return_value = mock_estimator

                estimator = create_optical_flow_estimator(model_type="auto", device="cpu")

                assert estimator == mock_estimator

    def test_create_unimatch_estimator_explicit(self):
        """Test factory creates UniMatch when explicitly requested."""
        with patch(
            "src.depth_surge_3d.models.optical_flow_estimator._try_load_unimatch"
        ) as mock_uni:
            mock_estimator = MagicMock()
            mock_uni.return_value = mock_estimator

            estimator = create_optical_flow_estimator(model_type="unimatch", device="cpu")

            assert estimator == mock_estimator
            mock_uni.assert_called_once_with("cpu", required=True)

    def test_create_raft_small_fallback(self):
        """Test factory falls back to RAFT-Small when large fails."""
        with patch(
            "src.depth_surge_3d.models.optical_flow_estimator._try_load_unimatch",
            return_value=None,
        ):
            with patch(
                "src.depth_surge_3d.models.optical_flow_estimator._try_load_raft_large",
                return_value=None,
            ):
                with patch(
                    "src.depth_surge_3d.models.optical_flow_estimator._try_load_raft_small"
                ) as mock_small:
                    mock_estimator = MagicMock()
                    mock_small.return_value = mock_estimator

                    estimator = create_optical_flow_estimator(model_type="auto", device="cpu")

                    assert estimator == mock_estimator

    def test_create_no_model_available(self):
        """Test factory raises error when no model available."""
        with patch(
            "src.depth_surge_3d.models.optical_flow_estimator._try_load_unimatch",
            return_value=None,
        ):
            with patch(
                "src.depth_surge_3d.models.optical_flow_estimator._try_load_raft_large",
                return_value=None,
            ):
                with patch(
                    "src.depth_surge_3d.models.optical_flow_estimator._try_load_raft_small",
                    return_value=None,
                ):
                    with pytest.raises(RuntimeError, match="No optical flow model available"):
                        create_optical_flow_estimator(model_type="auto", device="cpu")

    def test_device_auto_selection_cuda(self):
        """Test automatic device selection with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            estimator = RAFTFlowEstimator(device="auto")
            assert estimator.device == "cuda"

    def test_device_auto_selection_cpu(self):
        """Test automatic device selection falls back to CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                estimator = RAFTFlowEstimator(device="auto")
                assert estimator.device == "cpu"
