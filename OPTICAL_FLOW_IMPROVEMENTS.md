# Optical Flow Improvements

## Issues Addressed

### 1. ✅ Silent Fallback to RAFT
**Problem**: UniMatch fallback happened silently without warning
**Solution**: Added clear warning messages during auto-fallback

```
⚠️  UniMatch not available, falling back to RAFT...
✓ Using RAFT-Large as fallback optical flow model
```

### 2. ✅ No Visual Output
**Problem**: No way to verify optical flow is working
**Solution**: Saves intermediate outputs when `keep_intermediates=True`

**Output Structure**:
```
output/batch_name/
├── optical_flow/
│   ├── statistics.json                    # Quantitative metrics
│   ├── flow_visualizations/               # Colored flow maps
│   │   ├── flow_0000.png
│   │   ├── flow_0030.png
│   │   └── ...
│   └── depth_comparisons/                 # Side-by-side comparisons
│       ├── comparison_0000.png
│       ├── comparison_0030.png
│       └── ...
```

### 3. ✅ No Quantitative Metrics
**Problem**: No way to measure improvement
**Solution**: Detailed statistics output and JSON export

**Console Output**:
```
📊 Motion Compensation Statistics:
   Original temporal variance:     0.012345
   Compensated temporal variance:  0.008901
   Improvement:                    27.9%
   Mean absolute change per pixel: 0.003456
```

**JSON Export** (`statistics.json`):
```json
{
  "original_temporal_variance": 0.012345,
  "compensated_temporal_variance": 0.008901,
  "improvement_percentage": 27.9,
  "mean_absolute_change": 0.003456,
  "num_frames": 150
}
```

### 4. ✅ Better Logging Throughout
- Model selection clearly logged
- Fallback warnings with emoji indicators
- Progress updates for flow computation
- Intermediate save confirmation

## How to Test Improvements

### 1. Run with Optical Flow Enabled

```bash
python depth_surge_3d.py input.mp4 \
  --enable-optical-flow \
  --optical-flow-model auto \
  --optical-flow-blend 0.5 \
  --keep-intermediates
```

### 2. Check Output Directory

```bash
ls -la output/video_name_timestamp/optical_flow/
```

You should see:
- `statistics.json` - quantitative metrics
- `flow_visualizations/` - colored flow maps (HSV visualization)
- `depth_comparisons/` - original vs compensated depth maps

### 3. Analyze Flow Visualizations

Flow maps use HSV color encoding:
- **Hue (color)**: Flow direction (0°=right, 90°=down, 180°=left, 270°=up)
- **Saturation**: Always 100% (full color)
- **Value (brightness)**: Flow magnitude (brighter = faster motion)

### 4. Compare Depth Maps

Side-by-side comparisons show:
- **Left**: Original depth map (before compensation)
- **Right**: Compensated depth map (after optical flow)
- Look for smoother transitions between frames

### 5. Interpret Statistics

**Temporal Variance**:
- Measures frame-to-frame depth fluctuation
- Lower = more consistent/stable
- Target: 20-40% improvement

**Mean Absolute Change**:
- Average pixel-level change applied
- Too high (>0.01) = over-compensation risk
- Too low (<0.001) = minimal effect

## Optimal Blend Percentage Guide

### Finding the Sweet Spot

Test different blend values and compare:

```bash
# Conservative (20% optical flow)
python depth_surge_3d.py input.mp4 --enable-optical-flow --optical-flow-blend 0.2

# Balanced (50% optical flow) - DEFAULT
python depth_surge_3d.py input.mp4 --enable-optical-flow --optical-flow-blend 0.5

# Aggressive (80% optical flow)
python depth_surge_3d.py input.mp4 --enable-optical-flow --optical-flow-blend 0.8
```

Compare `improvement_percentage` in `statistics.json`:
- **Best**: 25-40% improvement with minimal artifacts
- **Too low**: <15% improvement (increase blend)
- **Too high**: >50% improvement but visible artifacts (decrease blend)

### Content-Specific Recommendations

| Content Type | Recommended Blend | Reasoning |
|-------------|------------------|-----------|
| Static camera | 0.6-0.8 | High confidence in flow accuracy |
| Panning camera | 0.4-0.6 | Moderate motion, balanced approach |
| Fast action | 0.2-0.4 | Reduce risk of motion blur artifacts |
| Scene cuts | 0.3-0.5 | Scene cut detection handles transitions |

## Verification Checklist

- [ ] See warning: "⚠️  UniMatch not available, falling back to RAFT..."
- [ ] See confirmation: "✓ Using RAFT-Large as fallback optical flow model"
- [ ] See detailed statistics output (4 metrics)
- [ ] Find `optical_flow/` directory in output
- [ ] Find `statistics.json` with valid metrics
- [ ] Find colorized flow visualizations
- [ ] Find side-by-side depth comparisons
- [ ] Improvement percentage is reasonable (15-50%)

## Example Session Output

```
Step 2.5/7: Applying optical flow motion compensation...
  -> Compensating 150 depth maps (blend=50.0%, model=auto)
⚠️  UniMatch not available, falling back to RAFT...
✓ Using RAFT-Large as fallback optical flow model
Loaded RAFT-LARGE on cuda
  -> Computing optical flow between frames...

  📊 Motion Compensation Statistics:
     Original temporal variance:     0.008234
     Compensated temporal variance:  0.005891
     Improvement:                    28.4%
     Mean absolute change per pixel: 0.002341

  💾 Saving optical flow intermediates to optical_flow/
     ✓ Saved 6 flow visualizations and depth comparisons

✓ Motion compensation complete in 45.23s
```

## Performance Impact

- **Processing Time**: +10-20% (flow computation + warping)
- **VRAM Usage**: +2-4GB (optical flow model)
- **Disk Space**: +50-200MB (intermediate visualizations)
  - Can disable with `--no-intermediates`

## Notes

- Flow visualizations sample every 30th frame (or 10% of frames)
- Depth comparisons use TURBO colormap (perceptually uniform)
- Statistics are computed on all frames, not just samples
- Scene cuts are automatically detected and skipped
