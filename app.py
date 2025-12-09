"""
Flask Backend for Pixel Revive - Image Restoration API
Connects the frontend to the image restoration processing
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'restored_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize DeOldify colorizer (lazy loading)
colorizer = None

def get_colorizer():
    global colorizer
    if colorizer is None:
        print("Loading DeOldify colorization model...")
        
        # Monkeypatch torch.load to use weights_only=False for PyTorch 2.6+
        import torch
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        from deoldify import visualize
        colorizer = visualize.get_image_colorizer(artistic=True, render_factor=35)
        
        # Restore original
        torch.load = original_load
        
        print("✅ Colorizer loaded!")
    return colorizer

def detect_and_remove_scratches(image_path, output_path):
    """Detect and remove scratches using intelligent adaptive inpainting"""
    try:
        print("🔍 Detecting and removing scratches (Smart Mode)...")
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Failed to read image")
            return False
        
        # Convert to grayscale with histogram equalization for better detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        
        # DETECT DARK SCRATCHES (blackhat)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        
        blackhat_v = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, kernel_v)
        blackhat_h = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, kernel_h)
        
        # DETECT WHITE SCRATCHES (tophat) - for severely damaged old photos
        tophat_v = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel_v)
        tophat_h = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel_h)
        
        # Combine all scratch types
        scratches = cv2.add(blackhat_v, blackhat_h)
        scratches = cv2.add(scratches, tophat_v)
        scratches = cv2.add(scratches, tophat_h)
        
        # Conservative threshold to avoid false positives
        _, mask = cv2.threshold(scratches, 22, 255, cv2.THRESH_BINARY)
        
        # Detect edges to protect facial boundaries
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Remove scratches that overlap with edges (protect face boundaries)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges_dilated))
        
        # Remove small noise (keep only significant scratches)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Very gentle dilation (minimal expansion)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Use smaller inpaint radius for more precise edges
        result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        # Apply very gentle bilateral filter (preserve more detail)
        result = cv2.bilateralFilter(result, d=5, sigmaColor=40, sigmaSpace=40)
        
        # Blend result with original using smooth mask (gentle application)
        mask_blur = cv2.GaussianBlur(mask.astype(float) / 255.0, (3, 3), 0)
        mask_3ch = np.stack([mask_blur] * 3, axis=-1)
        result = (result * mask_3ch + img * (1 - mask_3ch)).astype(np.uint8)
        
        # Save result
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("✅ Scratch removal complete (Smart adaptive mode)!")
        return True
    except Exception as e:
        print(f"⚠️ Scratch removal failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def enhance_quality(image_path, output_path):
    """Enhance image quality using advanced filters"""
    try:
        print("✨ Enhancing image quality...")
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Failed to read image")
            return False
        
        print("  → Denoising...")
        # Stronger denoising for old photos
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        
        print("  → Sharpening...")
        # Unsharp masking for better sharpening
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        print("  → Enhancing contrast...")
        # CLAHE for contrast enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        print("  → Adjusting brightness...")
        # Slight brightness adjustment
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 10)
        v = np.clip(v, 0, 255)
        enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
        
        # Save result with high quality
        cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("✅ Quality enhancement complete!")
        return True
    except Exception as e:
        print(f"⚠️ Quality enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the frontend HTML"""
    return send_from_directory('.', 'frontend.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files like videos"""
    if filename.endswith('.mp4'):
        return send_from_directory('.', filename)
    return '', 404

@app.route('/api/restore', methods=['POST'])
def restore_image():
    """
    API endpoint to restore/enhance uploaded image - Full Image Restoration + Colorization
    Accepts: multipart/form-data with 'image' file and optional 'colorize' parameter
    Returns: JSON with restored image path
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
        
        # Check processing options
        colorize = request.form.get('colorize', 'false').lower() == 'true'
        remove_scratches = request.form.get('remove_scratches', 'true').lower() == 'true'
        enhance = request.form.get('enhance_quality', 'true').lower() == 'true'
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        input_filename = f"{unique_id}_input.{ext}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        
        # Save uploaded file
        file.save(input_path)
        
        # Process image with FULL restoration pipeline (like demo.py)
        print(f"Processing full image restoration: {input_path}")
        
        # Create temporary directory for this processing
        temp_input = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
        temp_output = os.path.join(app.config['OUTPUT_FOLDER'], unique_id)
        os.makedirs(temp_input, exist_ok=True)
        os.makedirs(temp_output, exist_ok=True)
        
        # Preprocessing: Scratch removal and quality enhancement
        import shutil
        temp_img_path = os.path.join(temp_input, f"input.{ext}")
        processed_path = input_path
        
        processing_steps = []
        
        if remove_scratches:
            print("\n" + "="*50)
            print("STEP 1: Scratch Removal")
            print("="*50)
            scratch_removed_path = os.path.join(temp_input, f"scratch_removed.{ext}")
            if detect_and_remove_scratches(input_path, scratch_removed_path):
                processed_path = scratch_removed_path
                processing_steps.append("Scratch Removal")
                print("✅ Scratch removal applied successfully")
            else:
                print("⚠️ Scratch removal skipped (no changes needed)")
        
        if enhance:
            print("\n" + "="*50)
            print("STEP 2: Quality Enhancement")
            print("="*50)
            enhanced_path = os.path.join(temp_input, f"enhanced.{ext}")
            if enhance_quality(processed_path, enhanced_path):
                processed_path = enhanced_path
                processing_steps.append("Quality Enhancement")
                print("✅ Quality enhancement applied successfully")
            else:
                print("⚠️ Quality enhancement skipped")
        
        # Copy processed image for GPEN
        print("\n" + "="*50)
        print("STEP 3: GPEN Face Restoration")
        print("="*50)
        shutil.copy(processed_path, temp_img_path)
        processing_steps.append("GPEN Restoration")
        
        # Use GPEN directly instead of subprocess
        import sys
        import __init_paths  # Initialize paths
        from face_enhancement import FaceEnhancement
        
        # Initialize face enhancement
        print("Initializing FaceEnhancement with use_sr=False...")
        try:
            faceenhancer = FaceEnhancement(
                base_dir='./',
                in_size=512,
                model='GPEN-BFR-512',
                use_sr=False,
                sr_model=None,  # Don't load SR model when use_sr=False
                sr_scale=2,
                channel_multiplier=2,
                device='cpu'  # Use CPU for stability
            )
            print("✅ FaceEnhancement initialized successfully")
        except Exception as e:
            print(f"❌ FaceEnhancement initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Process all images in temp_input
        img_list = [f for f in os.listdir(temp_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in img_list:
            img_path = os.path.join(temp_input, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img, orig_faces, enhanced_faces = faceenhancer.process(img)
                
                # Save output
                img_name_only = os.path.splitext(img_name)[0]
                output_filename = f"{img_name_only}_GPEN.jpg"
                final_output_path = os.path.join(temp_output, output_filename)
                cv2.imwrite(final_output_path, img)
        
        print("✅ GPEN restoration complete!")
        
        # Verify output was created
        if not os.path.exists(final_output_path):
            return jsonify({'error': 'Face enhancement processing failed - no output generated'}), 500
        
        # Apply colorization if requested
        if colorize:
            print("\n" + "="*50)
            print("STEP 4: AI Colorization")
            print("="*50)
            try:
                col = get_colorizer()
                colorized_path = os.path.join(temp_output, f"{unique_id}_colorized.png")
                
                print("🎨 Colorizing image with DeOldify...")
                result_img = col.get_transformed_image(
                    Path(final_output_path),
                    render_factor=35,
                    post_process=True,
                    watermarked=False
                )
                if result_img is not None:
                    result_img.save(colorized_path, quality=95)
                    result_img.close()
                    output_filename = f"{unique_id}_colorized.png"
                    processing_steps.append("AI Colorization")
                    print("✅ Colorization complete!")
                else:
                    print("⚠️ Colorization returned None, using original restored image")
            except Exception as e:
                print(f"⚠️ Colorization failed: {e}")
                import traceback
                traceback.print_exc()
                print("Using restored image without colorization")
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print("="*50)
        print(f"Applied steps: {', '.join(processing_steps)}")
        print("="*50 + "\n")
        
        # Return result
        return jsonify({
            'success': True,
            'input_image': f'/uploads/{input_filename}',
            'restored_image': f'/output/{unique_id}/{output_filename}',
            'message': f'Image processed successfully! Applied: {", ".join(processing_steps)}',
            'processing_steps': processing_steps
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<unique_id>/<filename>')
def output_file(unique_id, filename):
    """Serve restored output files"""
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], unique_id), filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Pixel Revive API'})

if __name__ == '__main__':
    import sys
    print("=" * 60)
    print("🚀 Pixel Revive Backend Starting...")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("Backend API: http://localhost:5000")
    print("Frontend: http://localhost:5000")
    print("API Endpoint: http://localhost:5000/api/restore")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
