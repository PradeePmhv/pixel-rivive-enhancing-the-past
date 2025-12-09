# Image Restoration and Enhancement - Setup Guide

## Current Status

✅ **Completed:**
- All Python dependencies installed (torch, torchvision, fastai, etc.)
- Created `weights/` directory
- Downloading `global_checkpoints.zip` (for old photo restoration)

⏳ **In Progress:**
- Downloading checkpoints from Microsoft (about 526 MB)

❌ **Still Needed:**
- GPEN model weights (for face enhancement)
- DeOldify colorization models

## Quick Start Options

### Option 1: Download Models Manually

1. **For Old Photo Restoration** (currently downloading):
   - Download: https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
   - Extract to project root (creates `checkpoints/` folder)

2. **For Face Enhancement (GPEN models)**:
   
   Download these files and place in `weights/` folder:
   
   - RetinaFace-R50.pth (106 MB) - Face detection
   - ParseNet-latest.pth (85 MB) - Face parsing
   - GPEN-BFR-512.pth (350 MB) - Main face restoration model
   - model_ir_se50.pth (166 MB) - Face recognition
   - realesrnet_x2.pth (67 MB) - Super resolution 2x
   - realesrnet_x4.pth (67 MB) - Super resolution 4x

   **Alternative sources:**
   - Use ModelScope (Chinese model repository): `pip install "modelscope[cv]"`
   - Check facexlib releases: https://github.com/xinntao/facexlib/releases
   - GPEN releases: https://github.com/yangxy/GPEN/tree/main/weights

3. **For Colorization (DeOldify)**:
   - Models auto-download on first use
   - Or download from: https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth
   - Place in `models/` folder

### Option 2: Use ModelScope (Recommended for GPEN)

```python
# Install ModelScope
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# Run this Python code to download models:
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

portrait_enhancement = pipeline(
    Tasks.image_portrait_enhancement,
    model='damo/cv_gpen_image-portrait-enhancement-hires'
)

# This downloads models to ~/.cache/modelscope/hub/damo
# Copy the models from there to your weights/ folder
```

### Option 3: Use Pre-packaged Version

Download the complete package (4.5 GB) with all models included:
http://mizosoft.imagerestoration.pysimplegui.org

## Running the Project

### After Models are Downloaded:

1. **Simple face enhancement:**
   ```bash
   python demo.py --indir test_photos --outdir output
   ```

2. **Full pipeline with options:**
   ```bash
   python main.py --input_folder test_photos --output_folder output
   ```

3. **With all features:**
   ```bash
   python main.py --input_folder test_photos --output_folder output --sr_scale 4 --use_gpu --colorize
   ```

4. **GUI Application:**
   ```bash
   python gui.py
   ```

### Command Line Options:

- `--input_folder`: Input directory with old photos
- `--output_folder`: Where to save results
- `--sr_scale`: Super resolution scale (2 or 4)
- `--use_gpu`: Use GPU acceleration (if available)
- `--colorize`: Apply colorization to results
- `--inpaint_scratches`: Detect and repair scratches
- `--hr_quality`: High resolution quality enhancement
- `--run_mode`: 
  - 1 = ENHANCE_RESTORE (face first, then quality)
  - 2 = RESTORE_ENHANCE (quality first, then face)
  - 3 = ONLY_RESTORE
  - 4 = ONLY_ENHANCE

## Troubleshooting

### Missing Model Errors

If you see `FileNotFoundError` for model files:
1. Check the exact filename required in the error message
2. Ensure it's in the correct directory (`weights/` or `checkpoints/`)
3. Verify the file downloaded completely

### Out of Memory Errors

- Use `--tile_size 512` to process in tiles
- Reduce `--sr_scale` from 4 to 2
- Don't use `--use_gpu` if you have limited GPU memory

### Slow Processing

- Add `--use_gpu` if you have a CUDA-capable GPU
- Process fewer images at once
- Reduce output resolution

## File Structure

```
ImageRestorationAndEnhancement/
├── checkpoints/              # Old photo restoration models
│   └── restoration/
│       ├── VAE_A_quality/
│       ├── VAE_B_quality/
│       └── VAE_B_scratch/
├── weights/                  # GPEN face enhancement models
│   ├── RetinaFace-R50.pth
│   ├── ParseNet-latest.pth
│   ├── GPEN-BFR-512.pth
│   ├── model_ir_se50.pth
│   └── realesrnet_x4.pth
├── models/                   # DeOldify colorization models
│   └── ColorizeStable_gen.pth
├── test_photos/              # Input images
└── output/                   # Results
```

## Next Steps

1. Wait for `global_checkpoints.zip` to finish downloading
2. Extract it: `Expand-Archive global_checkpoints.zip .`
3. Download GPEN models (use download_models.py helper script)
4. Run the project!

## Helper Scripts Created

- `download_models.py` - Interactive script to download models
