# Depth Surge 3D - Future Improvements

## High Priority

### 🎥 Video-Optimized Depth Model
- [x] Integrate Video-Depth-Anything model ✓ (2025-10-12)
  - Repository: https://github.com/DepthAnything/Video-Depth-Anything
  - Created VideoDepthEstimator class for temporal consistency
  - Batch processor now uses video-aware depth estimation
  - Processes frames in 32-frame sliding windows with 10-frame overlap
- [ ] Download and configure video model weights
  - Need to download: video_depth_anything_vitl.pth (or smaller variants)
  - Update CLI to support --model-type flag (v2 vs video)
  - Add automatic model selection based on processing mode

### 🔄 Processing Mode Refactor
- [x] Clarify processing modes ✓ (2025-10-12)
  - Confirmed: "serial" = frame-by-frame with Depth Anything V2
  - Confirmed: "batch" = now implements true batch with video model support
  - Batch mode loads all frames, processes with temporal consistency
  - Clear architectural separation achieved
- [ ] Make batch mode the default (pending model download/testing)
- [ ] Update CLI help text to clarify mode differences
- [ ] Consider deprecating serial mode in future (after batch mode proven)

## Medium Priority

### ⚡ Performance Enhancements
- [ ] GPU memory optimization
  - Process frames in smaller batches to manage VRAM
  - Implement smart batch sizing based on available memory
- [ ] Parallel frame processing where possible
  - Depth estimation can be parallelized
  - Stereo generation can be parallelized
- [ ] Cache depth maps across runs
  - Skip re-computing depth if source video hasn't changed

### 🎨 UI/UX Improvements
- [ ] Real-time preview while processing
  - Show current frame being processed
  - Display depth map visualization
- [ ] Progress time estimates
  - Better ETA calculation based on actual frame processing time
- [ ] Drag-and-drop video upload
- [ ] Queue multiple videos for batch processing
- [ ] Resume interrupted processing automatically

### 🎛️ Advanced Settings
- [ ] Fine-tune stereo parameters per VR headset
  - Presets for Quest 2/3, Vive, PSVR2, etc.
- [ ] Custom depth model selection
  - Support multiple depth estimation models
  - Allow model switching without reinstall
- [ ] Advanced hole-filling algorithms
  - Depth-guided inpainting improvements
  - ML-based hole filling

## Low Priority

### 📦 Export Formats
- [ ] Additional VR-compatible formats
  - Oculus-specific formats
  - SteamVR-optimized output
- [ ] 360° video support
  - Equirectangular projection
  - Cubemap support

### 🧪 Experimental Features
- [ ] Dynamic depth adjustment
  - Auto-tune baseline/focal based on scene content
- [ ] Scene detection
  - Different 3D strength for different scene types
  - Cut detection to reset temporal models
- [ ] AI upscaling integration
  - Real-ESRGAN or similar for enhanced quality

### 📊 Analysis & Debugging
- [ ] Depth map quality metrics
  - Temporal consistency scoring
  - Edge coherence analysis
- [ ] Frame comparison tools
  - Side-by-side before/after viewer
  - Depth map visualization tools

## Documentation
- [ ] Update CLAUDE.md with current architecture
  - Document Video-Depth-Anything integration
  - Explain batch vs serial processing differences
  - Update model loading information
- [ ] Add processing mode comparison guide
  - Serial: frame-by-frame, lower memory, V2 model
  - Batch: temporal consistency, higher memory, video model
- [ ] Create VR headset compatibility matrix
- [ ] Document performance benchmarks by GPU
  - Compare V2 vs Video-Depth-Anything performance
  - Memory usage comparison

## Technical Debt
- [ ] Clean up commented debug code
- [ ] Standardize error handling patterns
- [ ] Add type hints throughout
- [ ] Comprehensive test suite
  - Unit tests for image processing
  - Integration tests for full pipeline
  - Performance regression tests

---

## Completed ✓

### 2025-10-12 Session
- [x] Integrated Video-Depth-Anything model for temporal consistency
- [x] Created VideoDepthEstimator class with batch processing support
- [x] Completely refactored BatchProcessor from delegation to true batch mode
- [x] Updated constants.py with video model configurations
- [x] Fixed .gitignore to properly handle model code vs model weights
- [x] Analyzed and clarified serial vs batch processing modes

### Previous Sessions
- [x] Fix Depth Anything V2 import issue
- [x] Test full pipeline with sample video
- [x] Update documentation to match current state
- [x] UI redesign - "Terminal Precision" aesthetic
- [x] Fix 'frames' KeyError with --no-intermediates
- [x] Workflow-guided button shininess
- [x] Dense grid background animation
