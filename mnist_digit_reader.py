import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify, render_template
import base64
import io
from scipy import ndimage

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DigitReader:
    def __init__(self):
        self.model = MNISTNet()
        self.load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def load_model(self):
        try:
            self.model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
            self.model.eval()
        except FileNotFoundError:
            print("Warning: Model file not found. Using default model.")
    
    def preprocess_image(self, image):
        # Convert to grayscale
        image = image.convert('L')
        
        # Invert colors (make digit white on black background)
        image = ImageOps.invert(image)
        
        # Apply threshold
        image = image.point(lambda x: 255 if x > 100 else 0)
        
        # Get the bounding box of the content
        bbox = image.getbbox()
        if bbox:
            # Add padding
            padding = 10
            bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image.size[0], bbox[2] + padding),
                min(image.size[1], bbox[3] + padding)
            )
            
            # Crop to the content
            image = image.crop(bbox)
            
            # Resize to 28x28 while maintaining aspect ratio
            target_size = 28
            ratio = min(target_size / image.size[0], target_size / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create final 28x28 image with centered content
            final_image = Image.new('L', (28, 28), 0)
            paste_x = (28 - image.size[0]) // 2
            paste_y = (28 - image.size[1]) // 2
            final_image.paste(image, (paste_x, paste_y))
            
            return final_image
        return Image.new('L', (28, 28), 0)
    
    def recognize_digit(self, image):
        # Preprocess the image
        processed_img = self.preprocess_image(image)
        
        # Convert to tensor and normalize
        img_tensor = self.transform(processed_img).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() > 0.1:  # Confidence threshold
                return str(predicted.item()), confidence.item()
        
        return None, 0.0

app = Flask(__name__)
reader = DigitReader()

@app.route('/')
def index():
    return render_template('digit_reader.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Recognize the digit
        digit, confidence = reader.recognize_digit(image)
        
        if digit is None:
            return jsonify({
                'success': False,
                'message': 'Could not recognize the digit. Please draw more clearly.',
                'digit': None,
                'confidence': 0.0
            })
        
        return jsonify({
            'success': True,
            'digit': digit,
            'confidence': confidence.item()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'digit': None,
            'confidence': 0.0
        })

if __name__ == '__main__':
    app.run(debug=True) 