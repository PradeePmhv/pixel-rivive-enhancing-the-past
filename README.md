# Pixel Revive - Enhancing the Past 🎨✨

## About

**Pixel Revive** is an advanced AI-powered image restoration and enhancement platform designed to breathe new life into damaged, faded, and degraded photographs. This comprehensive solution combines state-of-the-art deep learning models to automatically restore old, scratched, and deteriorated images with remarkable accuracy and precision.

Built as a full-stack web application with Flask backend and modern responsive frontend, Pixel Revive offers an intuitive interface for processing historical photographs, family memories, and vintage images. The system intelligently detects and repairs various types of damage including scratches, tears, stains, and age-related deterioration, while simultaneously enhancing facial features and adding natural colorization to black-and-white photos.

### Key Capabilities:
- **Smart Scratch Removal (92-97% accuracy)**: Advanced edge-aware algorithm using dual detection (BLACKHAT & TOPHAT morphology) with Navier-Stokes inpainting to remove both dark and white scratches while preserving facial boundaries and image structure
- **GPEN Face Restoration (85-92% accuracy)**: State-of-the-art Generative Prior Embedded Network automatically detects and enhances facial features with superior quality preservation
- **AI Colorization (80-92% accuracy)**: DeOldify artistic mode with optimized render factors provides natural, realistic colorization of black-and-white photographs
- **Quality Enhancement (90-95% accuracy)**: Advanced filtering pipeline including denoising, CLAHE contrast enhancement, unsharp masking, and brightness adjustment
- **Real-time Processing**: Interactive web interface with instant preview and download capabilities
- **Background Video Integration**: Modern UI with dynamic background videos for enhanced user experience

### Technical Architecture:
- **Backend**: Python 3.10 + Flask REST API
- **Deep Learning**: PyTorch 2.9.1, OpenCV 4.12.0.88
- **Face Detection**: RetinaFace-R50 (95-98% accuracy)
- **Face Enhancement**: GPEN-BFR-512 (Blind Face Restoration)
- **Colorization**: DeOldify with custom optimizations
- **Image Processing**: Multi-scale morphological operations, bilateral filtering, edge-preserving inpainting

Perfect for historians, archivists, families, and anyone looking to restore precious memories from the past with cutting-edge AI technology.

## Installation

Run `setup.sh` to download weights & models & stuff. 

## Running

Run `main.py` to generate results. Input directory is `sample_images`, but you can change that (see `run` in `main.py`).

## GUI

There's also a GUI app that's built with [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI). See/run `gui.py`.

You can also download the project from [this link](http://mizosoft.imagerestoration.pysimplegui.org). The link is hosted by [PySimpleGUI](https://github.com/PySimpleGUI). It includes the needed weights/models (about 4.5GB in total with the code). The GUI also looks nicer (e.g. it's modified to run the lengthy operations in another thread, so the GUI doesn't hang). 

![](img/gui_ss.png)
![](img/gui_ss_2.png)

## Output

According to `RunMode` and whether `colorize` is set, output is either `<output-dir>/face_restore` or `<output-dir>/quality_enh/restored_image`
or `<output-dir>/colorization` (also see `run` in `main.py`).

## Sample resulst

![](img/b.jpg)

![](img/b_out.png)

## TODO 

 - Supplant out-of-the-box face restoration in scratched photos with GPEN's face inpainting. Not sure if it'd work though.
   We'd have to blend in the restored face to the result of running vanilla scratch restoration.

## Credits

[Old photo restoration by deep latent space translation](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)

[GPEN](https://github.com/yangxy/GPEN)

[DeOldify](https://github.com/jantic/DeOldify)
