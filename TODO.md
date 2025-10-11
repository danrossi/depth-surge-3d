# Depth Surge 3D - Future Improvements

## High Priority

### 🎥 Video-Optimized Depth Model
- [ ] Replace Depth Anything V2 with Video-Depth-Anything
  - Repository: https://github.com/DepthAnything/Video-Depth-Anything
  - Benefits: Temporal consistency, better for video processing
  - May significantly improve quality and reduce flickering between frames

### 🔄 Processing Mode Refactor
- [ ] Clarify/rename processing modes
  - Current "serial" mode may actually be doing parallel work
  - "Batch" mode seems to have more advantages
  - Consider consolidating to single optimized batch-first approach
- [ ] Make batch mode the default
- [ ] Remove or deprecate serial mode if redundant

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
- [ ] Add processing mode comparison guide
- [ ] Create VR headset compatibility matrix
- [ ] Document performance benchmarks by GPU

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
- [x] Fix Depth Anything V2 import issue
- [x] Test full pipeline with sample video
- [x] Update documentation to match current state
- [x] UI redesign - "Terminal Precision" aesthetic
- [x] Fix 'frames' KeyError with --no-intermediates
- [x] Workflow-guided button shininess
- [x] Dense grid background animation
