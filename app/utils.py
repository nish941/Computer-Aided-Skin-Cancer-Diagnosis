import cv2
import numpy as np
from PIL import Image
import io

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]   # ImageNet std
    
    def preprocess(self, image_file):
        """
        Preprocess uploaded image for ResNet-50 model
        Steps: Grayscale -> Histogram Equalization -> Resize -> Normalize
        """
        try:
            # Read image from file object
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Convert to OpenCV format
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Convert to grayscale (as per your preprocessing)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            equalized = cv2.equalizeHist(gray)
            
            # Convert back to 3 channels (ResNet expects 3 channels)
            rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            
            # Resize to target size
            resized = cv2.resize(rgb, self.target_size)
            
            # Convert to float and normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization (expected by ResNet)
            for i in range(3):
                normalized[:, :, i] = (normalized[:, :, i] - self.mean[i]) / self.std[i]
            
            return normalized
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None
    
    def prepare_for_display(self, preprocessed_image):
        """
        Convert preprocessed image back to displayable format
        """
        try:
            # Denormalize
            denormalized = preprocessed_image.copy()
            for i in range(3):
                denormalized[:, :, i] = denormalized[:, :, i] * self.std[i] + self.mean[i]
            
            # Scale back to 0-255
            denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(denormalized)
            
            # Convert to base64 for HTML display
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"❌ Display preparation error: {e}")
            return None
