import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
import matplotlib.pyplot as plt
import io
import base64

class SkinCancerModel:
    def __init__(self, model_path='models/resnet50_model.h5'):
        """
        Load the trained ResNet-50 model for skin cancer classification
        """
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            
            # Class names based on ISIC 2018 dataset
            self.class_names = [
                'Melanoma (MEL)',
                'Nevus (NV)',
                'Basal Cell Carcinoma (BCC)',
                'Actinic Keratoses (AKIEC)',
                'Benign Keratosis (BKL)',
                'Dermatofibroma (DF)',
                'Vascular lesions (VASC)'
            ]
            
            # Medical information for each class
            self.class_info = {
                0: {
                    'name': 'Melanoma',
                    'abbr': 'MEL',
                    'severity': 'High',
                    'description': 'Most serious type of skin cancer. Requires immediate medical attention.',
                    'action': 'Consult a dermatologist immediately',
                    'common_locations': 'Face, chest, legs, back'
                },
                1: {
                    'name': 'Nevus',
                    'abbr': 'NV',
                    'severity': 'Low',
                    'description': 'Common mole, usually benign but monitor for changes.',
                    'action': 'Regular self-examination recommended',
                    'common_locations': 'Anywhere on body'
                },
                2: {
                    'name': 'Basal Cell Carcinoma',
                    'abbr': 'BCC',
                    'severity': 'Medium',
                    'description': 'Most common but least dangerous skin cancer. Rarely spreads.',
                    'action': 'Schedule dermatologist appointment',
                    'common_locations': 'Sun-exposed areas'
                },
                3: {
                    'name': 'Actinic Keratoses',
                    'abbr': 'AKIEC',
                    'severity': 'Medium-High',
                    'description': 'Pre-cancerous growths that can develop into SCC.',
                    'action': 'Consult dermatologist within 2-4 weeks',
                    'common_locations': 'Face, ears, scalp, hands'
                },
                4: {
                    'name': 'Benign Keratosis',
                    'abbr': 'BKL',
                    'severity': 'Low',
                    'description': 'Harmless skin growths, often called seborrheic keratosis.',
                    'action': 'No immediate action needed',
                    'common_locations': 'Chest, back, face'
                },
                5: {
                    'name': 'Dermatofibroma',
                    'abbr': 'DF',
                    'severity': 'Low',
                    'description': 'Benign fibrous nodule, usually harmless.',
                    'action': 'Monitor for changes',
                    'common_locations': 'Legs, arms'
                },
                6: {
                    'name': 'Vascular lesions',
                    'abbr': 'VASC',
                    'severity': 'Low-Medium',
                    'description': 'Blood vessel abnormalities, usually benign.',
                    'action': 'Consult if changing appearance',
                    'common_locations': 'Face, neck, upper body'
                }
            }
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, preprocessed_image):
        """
        Make prediction on preprocessed image
        Returns: predicted class, confidence, and all probabilities
        """
        try:
            # Add batch dimension
            image_batch = np.expand_dims(preprocessed_image, axis=0)
            
            # Get predictions
            predictions = self.model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class]) * 100
            
            # Get all probabilities
            probabilities = {}
            for i, prob in enumerate(predictions[0]):
                probabilities[self.class_names[i]] = float(prob) * 100
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None, None
    
    def generate_gradcam(self, image, predicted_class):
        """
        Generate Grad-CAM heatmap for model interpretability
        """
        try:
            # Create Grad-CAM object
            gradcam = Gradcam(self.model)
            
            # Function to modify model output for Grad-CAM
            def model_modifier(m):
                m.layers[-1].activation = tf.keras.activations.linear
            
            # Function to get output for specific class
            def loss(output):
                return output[:, predicted_class]
            
            # Generate heatmap
            cam = gradcam(loss, 
                         np.expand_dims(image, axis=0),
                         model_modifier=model_modifier,
                         penultimate_layer=-2)
            
            cam = normalize(cam[0])
            
            # Create visualization
            plt.figure(figsize=(10, 5))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 2, 2)
            plt.imshow(image, alpha=0.5)
            plt.imshow(cam, cmap='jet', alpha=0.5)
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return heatmap_base64
            
        except Exception as e:
            print(f"❌ Grad-CAM error: {e}")
            return None
    
    def get_class_info(self, class_idx):
        """Get detailed information for a predicted class"""
        if class_idx in self.class_info:
            info = self.class_info[class_idx]
            info['full_name'] = self.class_names[class_idx]
            return info
        return None
