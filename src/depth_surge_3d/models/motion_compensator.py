"""
Motion compensation for depth maps using optical flow.

This module provides depth refinement through optical flow-based warping and blending.
"""

from __future__ import annotations

import cv2
import numpy as np

from .optical_flow_estimator import OpticalFlowEstimator


class MotionCompensator:
    """Motion-compensated depth refinement using optical flow."""

    def __init__(self, optical_flow_estimator: OpticalFlowEstimator):
        """
        Initialize motion compensator.

        Args:
            optical_flow_estimator: Optical flow estimator instance
        """
        self.flow_estimator = optical_flow_estimator

    def warp_depth_with_flow(self, depth_map: np.ndarray, optical_flow: np.ndarray) -> np.ndarray:
        """
        Warp depth map using optical flow.

        Uses backward warping: for each pixel in target, find where it came from.

        Args:
            depth_map: Depth map [H, W] float32 normalized 0-1
            optical_flow: Flow field [H, W, 2] where [:,:,0]=x-flow, [:,:,1]=y-flow

        Returns:
            Warped depth map [H, W] float32
        """
        h, w = depth_map.shape

        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Apply flow to get source coordinates (backward warp)
        # Flow is [h, w, 2] where [:,:,0]=x_flow, [:,:,1]=y_flow
        source_x = x_coords - optical_flow[:, :, 0]
        source_y = y_coords - optical_flow[:, :, 1]

        # Warp depth map using cv2.remap
        warped_depth = cv2.remap(
            depth_map,
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return warped_depth

    def blend_depths(
        self,
        original: np.ndarray,
        compensated: np.ndarray,
        alpha: float,
        occlusion_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Blend original and motion-compensated depth.

        Args:
            original: Original depth map [H, W] float32
            compensated: Flow-warped depth map [H, W] float32
            alpha: Blend weight (0=original, 1=compensated)
            occlusion_mask: Optional mask for occluded regions [H, W] bool

        Returns:
            Blended depth map [H, W] float32
        """
        # Basic linear blend
        blended = (1.0 - alpha) * original + alpha * compensated

        # In occluded regions, prefer original depth
        if occlusion_mask is not None:
            blended = np.where(occlusion_mask, original, blended)

        # Ensure depth stays in valid range
        return np.clip(blended, 0.0, 1.0)

    def detect_scene_cuts(self, frames: np.ndarray, threshold: float = 30.0) -> list[int]:
        """
        Detect scene cuts where motion compensation shouldn't apply.

        Uses mean absolute difference between consecutive frames.

        Args:
            frames: Video frames [N, H, W, 3] uint8
            threshold: Intensity difference threshold for cut detection

        Returns:
            List of frame indices where cuts occur
        """
        if len(frames) < 2:
            return []

        cuts = []

        for i in range(1, len(frames)):
            # Compute mean absolute difference
            diff = np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
            mean_diff = np.mean(diff)

            # If difference exceeds threshold, mark as scene cut
            if mean_diff > threshold:
                cuts.append(i)

        return cuts

    def detect_occlusions(
        self,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
        threshold: float = 1.0,
    ) -> np.ndarray:
        """
        Detect occluded regions via forward-backward consistency check.

        Args:
            forward_flow: Flow from frame_t to frame_t+1 [H, W, 2]
            backward_flow: Flow from frame_t+1 to frame_t [H, W, 2]
            threshold: Consistency error threshold in pixels

        Returns:
            Occlusion mask [H, W] bool (True = occluded)
        """
        h, w = forward_flow.shape[:2]

        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Warp backward flow using forward flow
        target_x = x + forward_flow[:, :, 0]
        target_y = y + forward_flow[:, :, 1]

        warped_backward_flow = cv2.remap(
            backward_flow,
            target_x,
            target_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Check consistency: forward + backward should ≈ 0
        consistency_error = np.sqrt(
            (forward_flow[:, :, 0] + warped_backward_flow[:, :, 0]) ** 2
            + (forward_flow[:, :, 1] + warped_backward_flow[:, :, 1]) ** 2
        )

        # Mark pixels with large error as occluded
        occlusion_mask = consistency_error > threshold

        return occlusion_mask

    def compensate_depth_batch(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        blend_alpha: float = 0.5,
        detect_occlusions: bool = False,
        scene_cut_threshold: float = 30.0,
    ) -> np.ndarray:
        """
        Apply motion compensation to depth maps with blending.

        Args:
            frames: Video frames [N, H, W, 3] BGR uint8
            depth_maps: Original depth maps [N, H, W] float32 normalized 0-1
            blend_alpha: Blend weight (0=original, 1=flow-compensated)
            detect_occlusions: Whether to detect and handle occlusions
            scene_cut_threshold: Threshold for scene cut detection

        Returns:
            Compensated depth maps [N, H, W] float32
        """
        if len(frames) != len(depth_maps):
            raise ValueError("Number of frames must match number of depth maps")

        if len(frames) < 2:
            # Not enough frames for motion compensation
            return depth_maps.copy()

        # Detect scene cuts
        scene_cuts = self.detect_scene_cuts(frames, scene_cut_threshold)

        # Estimate optical flow between consecutive frames
        forward_flows = self.flow_estimator.estimate_flow_batch(frames)

        # Initialize compensated depth maps
        compensated_depths = []

        # First frame: no previous frame, keep original
        compensated_depths.append(depth_maps[0].copy())

        # Process remaining frames
        for i in range(1, len(depth_maps)):
            # Check if this frame is after a scene cut
            if i in scene_cuts:
                # Don't apply compensation after scene cut
                compensated_depths.append(depth_maps[i].copy())
                continue

            # Get forward flow from previous to current frame
            flow_forward = forward_flows[i - 1]

            # Warp previous compensated depth using forward flow
            warped_depth = self.warp_depth_with_flow(compensated_depths[i - 1], flow_forward)

            # Optional: Detect occlusions
            occlusion_mask = None
            if detect_occlusions and i + 1 < len(frames):
                # Need forward and backward flow
                flow_backward = self.flow_estimator.estimate_flow(frames[i], frames[i - 1])
                occlusion_mask = self.detect_occlusions(flow_forward, flow_backward, threshold=1.0)

            # Blend original and warped depth
            blended_depth = self.blend_depths(
                depth_maps[i], warped_depth, blend_alpha, occlusion_mask
            )

            compensated_depths.append(blended_depth)

        return np.array(compensated_depths)

    def compensate_depth_temporal(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        blend_alpha: float = 0.5,
        window_size: int = 3,
    ) -> np.ndarray:
        """
        Apply temporal motion compensation using multiple past frames.

        More advanced version that considers multiple previous frames for better
        temporal consistency.

        Args:
            frames: Video frames [N, H, W, 3] BGR uint8
            depth_maps: Original depth maps [N, H, W] float32 normalized 0-1
            blend_alpha: Blend weight for flow compensation
            window_size: Number of past frames to consider (1-5)

        Returns:
            Compensated depth maps [N, H, W] float32
        """
        # TODO: Implement multi-frame temporal aggregation
        # For now, fall back to single-frame compensation
        return self.compensate_depth_batch(frames, depth_maps, blend_alpha, detect_occlusions=False)

    def measure_temporal_consistency(self, depth_maps: np.ndarray) -> float:
        """
        Measure temporal consistency of depth maps.

        Lower values indicate better temporal consistency.

        Args:
            depth_maps: Depth maps [N, H, W] float32

        Returns:
            Mean frame-to-frame depth variance
        """
        if len(depth_maps) < 2:
            return 0.0

        differences = np.abs(np.diff(depth_maps, axis=0))
        return float(np.mean(differences))

    def get_compensation_stats(
        self, original_depths: np.ndarray, compensated_depths: np.ndarray
    ) -> dict[str, float]:
        """
        Get statistics comparing original and compensated depth maps.

        Args:
            original_depths: Original depth maps [N, H, W]
            compensated_depths: Compensated depth maps [N, H, W]

        Returns:
            Dictionary with comparison statistics
        """
        original_consistency = self.measure_temporal_consistency(original_depths)
        compensated_consistency = self.measure_temporal_consistency(compensated_depths)

        improvement = (
            (original_consistency - compensated_consistency) / original_consistency * 100.0
            if original_consistency > 0
            else 0.0
        )

        return {
            "original_temporal_variance": float(original_consistency),
            "compensated_temporal_variance": float(compensated_consistency),
            "improvement_percentage": float(improvement),
            "mean_absolute_change": float(np.mean(np.abs(original_depths - compensated_depths))),
        }
