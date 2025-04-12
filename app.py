from flask import Flask, render_template, request, jsonify, send_file
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import base64
import io
import os
import tempfile
from vit_mnist import ViT
import gtts
from playsound import playsound
import cv2
from datetime import datetime
from gtts import gTTS

app = Flask(__name__)

class CurrencyReader:
    def __init__(self):
        self.model = ViT(
            image_size=28,
            patch_size=4,
            num_classes=10,
            dim=256,
            depth=8,
            heads=8,
            mlp_dim=512,
            channels=1,
            dim_head=32,
            dropout=0.1,
            emb_dropout=0.1
        )
        checkpoint = torch.load('vit_mnist_best.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.languages = {
            'English': 'en',
            'Hindi': 'hi',
            'Marathi': 'mr',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Japanese': 'ja',
            'Chinese': 'zh-cn'
        }

    def preprocess_digit(self, image_array):
        # Convert to PIL Image if it's a numpy array
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array.astype('uint8'))
        else:
            image = image_array

        # Convert to grayscale
        image = image.convert('L')
        
        # Invert colors (make digit white on black background)
        image = ImageOps.invert(image)
        
        # Threshold the image
        image = image.point(lambda x: 255 if x > 128 else 0)
        
        # Get the bounding box of the digit
        bbox = image.getbbox()
        if bbox:
            # Crop to the digit
            image = image.crop(bbox)
            
            # Add padding
            padding = int(min(image.size) * 0.1)
            image = ImageOps.expand(image, border=padding, fill=0)
            
            # Resize while maintaining aspect ratio
            target_size = 20
            ratio = min(target_size / image.size[0], target_size / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create final 28x28 image with centered digit
            final_image = Image.new('L', (28, 28), 0)
            paste_x = (28 - image.size[0]) // 2
            paste_y = (28 - image.size[1]) // 2
            final_image.paste(image, (paste_x, paste_y))
            
            return final_image
        return Image.new('L', (28, 28), 0)

    def predict_digit(self, image_array):
        try:
            processed_img = self.preprocess_digit(image_array)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            img_tensor = transform(processed_img).unsqueeze(0)

            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Print debug information
                print(f"Predicted digit: {predicted.item()}, Confidence: {confidence.item()}")
                
                # Lower the confidence threshold for better recognition
                if confidence.item() > 0.01:  # Further reduced threshold
                    return str(predicted.item()), confidence.item()
                return None, 0.0
        except Exception as e:
            print(f"Error in predict_digit: {str(e)}")
            return None, 0.0

    def text_to_speech(self, text, language):
        try:
            tts = gTTS(text=text, lang=self.languages[language], slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

class CheckScanner:
    def __init__(self):
        self.min_width = 800
        self.min_height = 400
        
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (should be the check)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Ensure minimum size
            if w >= self.min_width and h >= self.min_height:
                # Crop to check region
                check = image[y:y+h, x:x+w]
                
                # Apply perspective transform if needed
                check = self.align_check(check)
                
                return check
        return None
    
    def align_check(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get the corners
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            corners = cv2.approxPolyDP(max_contour, epsilon, True)
            
            if len(corners) == 4:
                # Sort corners in order: top-left, top-right, bottom-right, bottom-left
                corners = self.order_points(corners)
                
                # Get width and height
                width = int(max(
                    np.linalg.norm(corners[1] - corners[0]),
                    np.linalg.norm(corners[3] - corners[2])
                ))
                height = int(max(
                    np.linalg.norm(corners[2] - corners[1]),
                    np.linalg.norm(corners[3] - corners[0])
                ))
                
                # Define destination points
                dst_points = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Apply perspective transform
                matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
                aligned = cv2.warpPerspective(image, matrix, (width, height))
                
                return aligned
        return image
    
    def order_points(self, corners):
        # Initialize a list of coordinates
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left will have the smallest sum
        # Bottom-right will have the largest sum
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        
        # Top-right will have the smallest difference
        # Bottom-left will have the largest difference
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        
        return rect

reader = CurrencyReader()
scanner = CheckScanner()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        num_digits = int(data['num_digits'])
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        img_array = np.array(image)
        
        # Check if image is empty (all black)
        if np.all(img_array == 0):
            return jsonify({
                'success': False,
                'message': 'No digits detected. Please draw some digits first.',
                'partial_result': None
            })
        
        # For single digit, use the entire image
        if num_digits == 1:
            # Invert the image for better digit detection
            img_array = 255 - img_array
            digit, conf = reader.predict_digit(img_array)
            if digit is not None:
                return jsonify({
                    'success': True,
                    'amount': str(digit),
                    'confidence_scores': [conf]
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Please write the digit more clearly',
                    'partial_result': '?',
                    'confidence_scores': [0.0]
                })
        
        # For multiple digits, split the image
        digit_width = img_array.shape[1] // num_digits
        predictions = []
        confidences = []
        
        for i in range(num_digits):
            start_x = i * digit_width
            end_x = (i + 1) * digit_width
            digit_section = img_array[:, start_x:end_x]
            
            # Invert each digit section
            digit_section = 255 - digit_section
            
            # Predict each digit
            digit, conf = reader.predict_digit(digit_section)
            if digit is not None:
                predictions.append(digit)
                confidences.append(conf)
            else:
                predictions.append("?")
                confidences.append(0.0)
        
        result = ''.join(predictions)
        if "?" not in predictions:
            return jsonify({
                'success': True,
                'amount': result,
                'confidence_scores': confidences
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Please write digits more clearly',
                'partial_result': ' '.join(predictions),
                'confidence_scores': confidences
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'partial_result': None
        })

@app.route('/speak')
def speak():
    try:
        text = request.args.get('text', '')
        language = request.args.get('language', 'English')
        
        if not text:
            return jsonify({'success': False, 'message': 'No text provided'})
        
        # Generate speech
        audio_file = reader.text_to_speech(text, language)
        
        if audio_file:
            return send_file(audio_file, mimetype='audio/mp3')
        else:
            return jsonify({'success': False, 'message': 'Failed to generate speech'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/scan', methods=['POST'])
def scan_check():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        num_digits = int(data.get('num_digits', 1))
        language = data.get('language', 'English')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Check if this is a digit drawing or a check image
        if num_digits > 0:  # This is a digit drawing
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # For single digit, use the entire image
            if num_digits == 1:
                # Invert the image for better digit detection
                gray = 255 - gray
                digit, conf = reader.predict_digit(gray)
                if digit is not None:
                    # Create complete sentence based on language
                    amount_text = get_amount_text(digit, language)
                    return jsonify({
                        'success': True,
                        'amount': str(digit),
                        'amount_text': amount_text,
                        'confidence_scores': [conf],
                        'language': language
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Please write the digit more clearly',
                        'partial_result': '?',
                        'confidence_scores': [0.0]
                    })
            
            # For multiple digits, split the image
            digit_width = gray.shape[1] // num_digits
            predictions = []
            confidences = []
            
            for i in range(num_digits):
                start_x = i * digit_width
                end_x = (i + 1) * digit_width
                digit_section = gray[:, start_x:end_x]
                
                # Invert each digit section
                digit_section = 255 - digit_section
                
                # Predict each digit
                digit, conf = reader.predict_digit(digit_section)
                if digit is not None:
                    predictions.append(digit)
                    confidences.append(conf)
                else:
                    predictions.append("?")
                    confidences.append(0.0)
            
            result = ''.join(predictions)
            if "?" not in predictions:
                # Create complete sentence based on language
                amount_text = get_amount_text(result, language)
                return jsonify({
                    'success': True,
                    'amount': result,
                    'amount_text': amount_text,
                    'confidence_scores': confidences,
                    'language': language
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Please write digits more clearly',
                    'partial_result': ' '.join(predictions),
                    'confidence_scores': confidences
                })
        
        # If not a digit drawing, process as a check
        processed_image = scanner.preprocess_image(image)
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'message': 'Could not detect check in the image. Please try again.'
            })
        
        # Convert processed image back to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_data = base64.b64encode(buffer).decode('utf-8')
        
        # Generate speech for the amount
        amount_text = "The amount is 1000 rupees"  # Replace with actual amount
        audio_file = reader.text_to_speech(amount_text, language)
        
        if audio_file:
            # Play the audio
            try:
                playsound(audio_file)
                # Clean up the temporary file
                os.unlink(audio_file)
            except Exception as e:
                print(f"Error playing audio: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': 'Check detected and processed successfully.',
            'image': f'data:image/jpeg;base64,{processed_image_data}',
            'amount': amount_text
        })
        
    except Exception as e:
        print(f"Error in scan_check: {str(e)}")  # Add debug logging
        return jsonify({
            'success': False,
            'message': str(e)
        })

def get_amount_text(amount, language):
    """Generate the complete sentence for the amount in the specified language."""
    amount_texts = {
        'English': f"The amount is {amount} rupees",
        'Hindi': f"राशि {amount} रुपये ",
        'Marathi': f"रक्कम {amount} रुपये",
        'Spanish': f"La cantidad es {amount} rupias",
        'French': f"Le montant est de {amount} roupies",
        'German': f"Der Betrag ist {amount} Rupien",
        'Japanese': f"金額は{amount}ルピーです",
        'Chinese': f"金额是{amount}卢比"
    }
    return amount_texts.get(language, amount_texts['English'])

if __name__ == '__main__':
    app.run(debug=True) 