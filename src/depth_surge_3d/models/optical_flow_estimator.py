"""
Optical flow estimation for motion-compensated depth refinement.

This module provides optical flow estimators using RAFT and UniMatch models
for improved temporal consistency in depth maps.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
import torch


class OpticalFlowEstimator(ABC):
    """Base class for optical flow estimation."""

    def __init__(self, device: str = "auto"):
        """
        Initialize optical flow estimator.

        Args:
            device: Device to use (auto, cuda, cpu, mps)
        """
        self.device = self._determine_device(device)
        self.model = None
        self.model_type = "unknown"

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the optical flow model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    def estimate_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Estimate optical flow between two frames.

        Args:
            frame1: First frame [H, W, 3] BGR uint8
            frame2: Second frame [H, W, 3] BGR uint8

        Returns:
            Flow field [H, W, 2] float32 where [:,:,0]=x-flow, [:,:,1]=y-flow
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess frames
        img1 = self._preprocess_frame(frame1)
        img2 = self._preprocess_frame(frame2)

        # Add batch dimension
        img1_batch = img1.unsqueeze(0)
        img2_batch = img2.unsqueeze(0)

        # Estimate flow
        with torch.no_grad():
            flow = self._run_inference(img1_batch, img2_batch)

        # Convert to numpy
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow_np

    def estimate_flow_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        Estimate optical flow for consecutive frame pairs.

        Args:
            frames: Video frames [N, H, W, 3] BGR uint8

        Returns:
            Flow fields [N-1, H, W, 2] float32
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to estimate flow")

        flows = []
        for i in range(len(frames) - 1):
            flow = self.estimate_flow(frames[i], frames[i + 1])
            flows.append(flow)

        return np.array(flows)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for optical flow model.

        Args:
            frame: Frame [H, W, 3] BGR uint8

        Returns:
            Preprocessed tensor [3, H, W] float32 in [-1, 1]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to float and normalize to [0, 1]
        frame_float = frame_rgb.astype(np.float32) / 255.0

        # Convert to tensor [H, W, 3] -> [3, H, W]
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1)

        # Normalize to [-1, 1] for RAFT
        tensor = tensor * 2.0 - 1.0

        # Move to device
        tensor = tensor.to(self.device)

        return tensor

    @abstractmethod
    def _run_inference(self, img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> torch.Tensor:
        """
        Run model inference on preprocessed images.

        Args:
            img1_batch: First image batch [B, 3, H, W]
            img2_batch: Second image batch [B, 3, H, W]

        Returns:
            Flow field [B, 2, H, W]
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "loaded": self.model is not None,
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


class RAFTFlowEstimator(OpticalFlowEstimator):
    """RAFT optical flow estimator using torchvision."""

    def __init__(self, device: str = "auto", model_size: str = "large"):
        """
        Initialize RAFT estimator.

        Args:
            device: Device to use (auto, cuda, cpu, mps)
            model_size: Model size (large or small)
        """
        super().__init__(device)
        self.model_size = model_size
        self.model_type = f"raft_{model_size}"

    def load_model(self) -> bool:
        """
        Load RAFT model from torchvision.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from torchvision.models.optical_flow import (
                Raft_Large_Weights,
                Raft_Small_Weights,
                raft_large,
                raft_small,
            )

            print(f"Loading RAFT optical flow model ({self.model_size})...")

            # Load model based on size
            if self.model_size == "small":
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights, progress=True)
            else:  # large (default)
                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights, progress=True)

            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"Loaded RAFT-{self.model_size.upper()} on {self.device}")
            return True

        except ImportError as e:
            print(f"Error: torchvision optical flow models not available: {e}")
            print("Install/upgrade torchvision: pip install --upgrade torchvision")
            return False
        except Exception as e:
            print(f"Error loading RAFT model: {e}")
            return False

    def _run_inference(self, img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> torch.Tensor:
        """
        Run RAFT inference.

        Args:
            img1_batch: First image batch [B, 3, H, W]
            img2_batch: Second image batch [B, 3, H, W]

        Returns:
            Flow field [B, 2, H, W]
        """
        # RAFT returns list of flow predictions (12 iterations)
        # We use the final (most refined) prediction
        list_of_flows = self.model(img1_batch, img2_batch)
        final_flow = list_of_flows[-1]
        return final_flow


class UniMatchFlowEstimator(OpticalFlowEstimator):
    """UniMatch optical flow estimator."""

    def __init__(self, device: str = "auto"):
        """
        Initialize UniMatch estimator.

        Args:
            device: Device to use (auto, cuda, cpu, mps)
        """
        super().__init__(device)
        self.model_type = "unimatch"

    def load_model(self) -> bool:
        """
        Load UniMatch model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Suppress warnings during import
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try to import UniMatch
                # This will only work if the package is installed
                import unimatch  # noqa: F401

            print("Loading UniMatch optical flow model...")

            # TODO: Implement UniMatch loading
            # This is a placeholder for now
            print("UniMatch implementation pending - falling back to RAFT")
            return False

        except ImportError:
            # UniMatch not installed - this is expected fallback behavior
            return False
        except Exception as e:
            print(f"Error loading UniMatch model: {e}")
            return False

    def _run_inference(self, img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> torch.Tensor:
        """
        Run UniMatch inference.

        Args:
            img1_batch: First image batch [B, 3, H, W]
            img2_batch: Second image batch [B, 3, H, W]

        Returns:
            Flow field [B, 2, H, W]
        """
        # TODO: Implement UniMatch inference
        raise NotImplementedError("UniMatch inference not yet implemented")


def _try_load_unimatch(device: str, required: bool) -> OpticalFlowEstimator | None:
    """Try to load UniMatch model."""
    try:
        estimator = UniMatchFlowEstimator(device)
        if estimator.load_model():
            print("Using UniMatch optical flow")
            return estimator
    except Exception as e:
        if required:
            print(f"Warning: UniMatch requested but unavailable: {e}")
    return None


def _try_load_raft_large(device: str) -> OpticalFlowEstimator | None:
    """Try to load RAFT-Large model."""
    try:
        estimator = RAFTFlowEstimator(device, model_size="large")
        if estimator.load_model():
            print("Using RAFT-Large optical flow")
            return estimator
    except Exception as e:
        print(f"RAFT-Large unavailable: {e}")
    return None


def _try_load_raft_small(device: str) -> OpticalFlowEstimator | None:
    """Try to load RAFT-Small model."""
    try:
        estimator = RAFTFlowEstimator(device, model_size="small")
        if estimator.load_model():
            print("Using RAFT-Small optical flow")
            return estimator
    except Exception as e:
        print(f"RAFT-Small unavailable: {e}")
    return None


def create_optical_flow_estimator(
    model_type: str = "auto", device: str = "auto"
) -> OpticalFlowEstimator:
    """
    Factory function to create optical flow estimator with automatic fallback.

    Args:
        model_type: Model type (auto, unimatch, raft, raft_small)
        device: Device for inference (auto, cuda, cpu, mps)

    Returns:
        Configured optical flow estimator

    Raises:
        RuntimeError: If no optical flow model is available
    """
    # Try UniMatch first if auto or explicitly requested
    if model_type in ("auto", "unimatch"):
        estimator = _try_load_unimatch(device, required=(model_type == "unimatch"))
        if estimator is not None:
            return estimator
        # Warn about fallback when auto mode
        if model_type == "auto":
            print("⚠️  UniMatch not available, falling back to RAFT...")

    # Try RAFT-Large
    if model_type in ("auto", "raft", "raft_large"):
        estimator = _try_load_raft_large(device)
        if estimator is not None:
            if model_type == "auto":
                print("✓ Using RAFT-Large as fallback optical flow model")
            return estimator
        # Warn about further fallback
        if model_type == "auto":
            print("⚠️  RAFT-Large failed, trying RAFT-Small...")

    # Try RAFT-Small as final fallback
    if model_type in ("auto", "raft_small"):
        estimator = _try_load_raft_small(device)
        if estimator is not None:
            if model_type == "auto":
                print("✓ Using RAFT-Small as final fallback optical flow model")
            return estimator

    # No model available
    raise RuntimeError(
        "No optical flow model available. "
        "Ensure torchvision is installed: pip install --upgrade torchvision"
    )
