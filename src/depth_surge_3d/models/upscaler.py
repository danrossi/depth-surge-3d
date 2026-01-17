"""
AI upscaling models for output enhancement.

This module provides abstraction for various upscaling models including Real-ESRGAN,
with support for style-preserving enhancement without altering video aesthetics.
"""

from __future__ import annotations
from typing import Literal
import torch
import numpy as np
from PIL import Image

UpscaleModel = Literal["none", "x2", "x4", "x4-conservative"]


class ImageUpscaler:
    """Base class for AI upscaling models."""

    def __init__(self, device: str = "auto"):
        """
        Initialize upscaler.

        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._determine_device(device)
        self.model = None

    def _determine_device(self, device: str) -> str:
        """Determine best device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def load_model(self) -> bool:
        """
        Load the upscaling model.

        Returns:
            True if model loaded successfully
        """
        raise NotImplementedError

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale a single image.

        Args:
            image: Input image in BGR format (OpenCV convention)

        Returns:
            Upscaled image in BGR format
        """
        raise NotImplementedError

    def unload_model(self) -> None:
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


class RealESRGANUpscaler(ImageUpscaler):
    """Real-ESRGAN upscaler implementation."""

    def __init__(self, model_name: str = "x4", device: str = "auto"):
        """
        Initialize Real-ESRGAN upscaler.

        Args:
            model_name: Model variant ('x2', 'x4', 'x4-conservative')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        super().__init__(device)
        self.model_name = model_name
        self.scale = int(model_name[1]) if model_name.startswith("x") else 4

    def load_model(self) -> bool:
        """Load Real-ESRGAN model."""
        try:
            from py_real_esrgan.model import RealESRGAN

            # Map model names to weights
            weights_map = {
                "x2": "weights/RealESRGAN_x2plus.pth",
                "x4": "weights/RealESRGAN_x4plus.pth",
                "x4-conservative": "weights/RealESRNet_x4plus.pth",
            }

            weights_path = weights_map.get(self.model_name, weights_map["x4"])

            # Create model and load weights (auto-downloads if missing)
            self.model = RealESRGAN(self.device, scale=self.scale)
            self.model.load_weights(weights_path, download=True)

            print(f"Loaded Real-ESRGAN ({self.model_name}) on {self.device}")
            return True

        except Exception as e:
            print(f"Error loading Real-ESRGAN: {e}")
            print("Ensure py-real-esrgan is installed: pip install py-real-esrgan")
            return False

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale a single image (BGR → BGR).

        Args:
            image: Input image in BGR format

        Returns:
            Upscaled image in BGR format
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert BGR to RGB, numpy to PIL
        image_rgb = image[..., ::-1]  # BGR to RGB
        pil_image = Image.fromarray(image_rgb)

        # Upscale
        upscaled_pil = self.model.predict(pil_image)

        # Convert back to BGR numpy
        upscaled_rgb = np.array(upscaled_pil)
        upscaled_bgr = upscaled_rgb[..., ::-1]  # RGB to BGR

        return upscaled_bgr


def create_upscaler(model_name: str = "none", device: str = "auto") -> ImageUpscaler | None:
    """
    Factory function to create an upscaler.

    Args:
        model_name: Upscale model ('none', 'x2', 'x4', 'x4-conservative')
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        ImageUpscaler instance or None if model_name is 'none'

    Raises:
        ValueError: If model_name is unknown
    """
    if model_name == "none":
        return None

    # Real-ESRGAN models
    if model_name in ["x2", "x4", "x4-conservative"]:
        return RealESRGANUpscaler(model_name, device)

    raise ValueError(f"Unknown upscale model: {model_name}")
