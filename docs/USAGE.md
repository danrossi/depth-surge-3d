# Usage Guide

## Web UI (Recommended)

Launch the web interface for an intuitive experience:

```bash
./run_ui.sh
```

Then open http://localhost:5000 in your browser. The web UI provides:
- **Drag & drop video upload** with instant preview
- **Visual time range selection** with duration display
- **Real-time progress tracking** with frame previews
- **Interactive settings panel** with proper defaults
- **Batch management** with timestamped outputs
- **Live preview** of depth maps and stereo frames

## Quick Start Script

For command-line processing with time ranges:

```bash
# Place your video as 'input_video.mp4' in the project directory
# Then run with start and end times
./start.sh 1:11 2:22

# The script automatically:
# - Uses input_video.mp4 as source
# - Extracts the specified time range
# - Preserves audio in the output (by default)
# - Upscales to minimum 1080p resolution
# - Interpolates frames to 60fps for smooth VR experience
# - Uses optimized settings for high-quality processing
```

## Basic Command Line Usage

```bash
# With uv (recommended - runs directly without manual activation)
uv run python depth_surge_3d.py input_video.mp4

# Or activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then run the script
python depth_surge_3d.py input_video.mp4
```

## Advanced Command Line Usage

```bash
# Specify output directory and format
python depth_surge_3d.py input_video.mp4 -o ./my_output -f over_under

# Process only a specific time range
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 03:45

# Process with hours:minutes:seconds format
python depth_surge_3d.py input_video.mp4 -s 00:01:30 -e 00:03:45

# Process from start to specific time
python depth_surge_3d.py input_video.mp4 -e 02:00

# Process from specific time to end
python depth_surge_3d.py input_video.mp4 -s 01:00

# Adjust stereo parameters
python depth_surge_3d.py input_video.mp4 -b 0.15 -fl 1200

# Use CPU only (if no GPU available)
python depth_surge_3d.py input_video.mp4 --device cpu

# Don't save intermediate files
python depth_surge_3d.py input_video.mp4 --no-intermediates

# Disable audio preservation
python depth_surge_3d.py input_video.mp4 --no-audio

# Custom framerate and resolution
python depth_surge_3d.py input_video.mp4 --fps 120 --resolution 4k

# Combined example: high-quality 4K 60fps processing
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 02:00 -f over_under --resolution 4k --fps 60 --no-intermediates
```

## Output Structure

When processing a video, the tool creates a timestamped directory with all intermediate and final files:

```
output/
└── timestamp_videoname_timestamp/
    ├── original_video.mp4      # Source video (uploaded or copied)
    ├── original_audio.flac     # Pre-extracted lossless audio
    ├── frames/                 # Extracted video frames
    ├── depth_maps/            # Generated depth maps (if --keep-intermediates)
    ├── left_frames/           # Left stereo images (if --keep-intermediates)
    ├── right_frames/          # Right stereo images (if --keep-intermediates)
    ├── vr_frames/             # Combined VR frames
    └── videoname_3D_side_by_side_0118-0133.mp4  # Final VR video
```

## Resume Interrupted Processing

If processing is interrupted (power failure, system crash, manual stop), you can resume from where it left off:

```bash
# Resume processing from any output directory
python depth_surge_3d.py --resume ./output/my_video_1234567890/

# The web UI automatically populates the resume field after upload
```

**Resume Process:**
1. **Validates directory**: Checks for valid settings file and intermediate files
2. **Analyzes progress**: Determines which processing steps were already completed
3. **Loads original settings**: Uses exact same parameters as the original run
4. **Continues processing**: Skips completed steps and continues from where it stopped

**Resume Capability:**
- ✅ **Step-level resume**: Skips entire completed steps (frame extraction, depth maps, stereo pairs, etc.)
- ✅ **Smart detection**: Automatically finds existing intermediate files
- ✅ **Progress analysis**: Shows exactly how much work was completed
- ❌ **Not needed**: Already completed processing

## VR Viewing

The generated video can be viewed with:
- VR headsets (Oculus, HTC Vive, etc.)
- Cardboard VR viewers
- 3D video players that support side-by-side or over-under formats
- Wide-screen displays (for cinema formats)

## Workflow Example

Here's a typical workflow:

1. **Setup** (one time):
   ```bash
   ./setup.sh
   ```

2. **Test setup** (optional but recommended):
   ```bash
   ./test.sh
   ```

3. **Prepare your video**:
   ```bash
   cp my_video.mp4 input_video.mp4
   ```

4. **Quick processing**:
   ```bash
   # Process a 15-second clip from 1:00 to 1:15 with audio
   ./start.sh 1:00 1:15
   ```

5. **Find your output**:
   ```bash
   # The processed video will be in the output/ directory
   ls -lh output/*/
   ```
