from flask import Flask, render_template, request, jsonify
import base64
import os
from datetime import datetime
from model_loader import SkinCancerModel
from utils import ImagePreprocessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'skin-cancer-diagnosis-secret-key'

# Initialize model and preprocessor
try:
    model = SkinCancerModel()
    preprocessor = ImagePreprocessor()
    print("✅ Application initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize application: {e}")
    model = None
    preprocessor = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    if '.' not in file.filename or \
       file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Preprocess image
        preprocessed_image = preprocessor.preprocess(file)
        
        if preprocessed_image is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        # Make prediction
        predicted_class, confidence, probabilities = model.predict(preprocessed_image)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Generate Grad-CAM heatmap
        heatmap_base64 = model.generate_gradcam(preprocessed_image, predicted_class)
        
        # Prepare image for display
        display_image = preprocessor.prepare_for_display(preprocessed_image)
        
        # Get class information
        class_info = model.get_class_info(predicted_class)
        
        # Format probabilities for display
        formatted_probs = {}
        for class_name, prob in probabilities.items():
            formatted_probs[class_name] = f"{prob:.2f}%"
        
        # Sort probabilities for chart
        sorted_probs = dict(sorted(probabilities.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'class_name': model.class_names[predicted_class],
                'confidence': f"{confidence:.2f}%",
                'raw_confidence': confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'probabilities': formatted_probs,
            'sorted_probabilities': sorted_probs,
            'class_info': class_info,
            'display_image': display_image,
            'heatmap': heatmap_base64
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/class_info/<int:class_id>')
def get_class_info(class_id):
    """Get detailed information about a specific class"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 0 <= class_id < len(model.class_names):
        info = model.get_class_info(class_id)
        return jsonify(info)
    else:
        return jsonify({'error': 'Invalid class ID'}), 404

@app.route('/about')
def about():
    """About page with project information"""
    project_info = {
        'title': 'Computer-Aided Skin Cancer Diagnosis',
        'description': 'A deep learning project using ResNet-50 for automated skin lesion classification',
        'accuracy': '84.3%',
        'sensitivity': '91%',
        'classes': len(model.class_names) if model else 7,
        'model': 'ResNet-50',
        'dataset': 'ISIC 2018 Challenge Dataset'
    }
    return render_template('about.html', info=project_info)

@app.route('/api/health')
def health_check():
    """Health check endpoint for deployment"""
    if model is not None and preprocessor is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    print("\n" + "="*50)
    print("🚀 Skin Cancer Diagnosis Web App")
    print("="*50)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🔗 Local URL: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
