"""
AI upscaling models for output enhancement.

This module provides abstraction for various upscaling models including Real-ESRGAN,
with support for style-preserving enhancement without altering video aesthetics.
"""

from __future__ import annotations
from typing import Literal
import torch
import numpy as np

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
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # Map model names to parameters
            if self.model_name == "x2":
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
                )
                netscale = 2
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            elif self.model_name == "x4-conservative":
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
                )
                netscale = 4
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            else:  # x4 default
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
                )
                netscale = 4
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

            # Determine device string for RealESRGANer
            device_str = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

            # Create upsampler
            self.model = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=0,  # No tiling (process full image)
                tile_pad=10,
                pre_pad=0,
                half=True if device_str == "cuda" else False,  # FP16 on GPU
                device=device_str,
            )

            print(f"Loaded Real-ESRGAN ({self.model_name}) on {device_str}")
            return True

        except Exception as e:
            print(f"Error loading Real-ESRGAN: {e}")
            print("Ensure realesrgan is installed: pip install realesrgan")
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

        # RealESRGANer.enhance expects BGR and returns BGR, face_enhance=False
        output, _ = self.model.enhance(image, outscale=self.scale)

        return output


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
