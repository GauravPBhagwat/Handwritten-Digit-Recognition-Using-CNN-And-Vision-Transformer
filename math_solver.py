import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import re
from scipy import ndimage

class MathSolver:
    def __init__(self):
        # Initialize the Vision Transformer model for character recognition
        self.model = self._create_model()
        self.load_model()
        
        # Define mathematical operators and symbols
        self.operators = ['+', '-', '*', '/', '=', '(', ')', '^']
        self.symbols = ['x', 'y', 'z']
        
        # Initialize SymPy symbols
        self.sympy_symbols = {symbol: sympy.Symbol(symbol) for symbol in self.symbols}
        
    def _create_model(self):
        # Create a Vision Transformer model for character recognition
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 36)  # 10 digits + 26 operators/symbols
        )
        return model
    
    def load_model(self):
        try:
            checkpoint = torch.load('math_solver_model.pth', map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except FileNotFoundError:
            print("Warning: Model file not found. Using default model.")
    
    def preprocess_image(self, image):
        # Convert to grayscale
        image = image.convert('L')
        
        # Invert colors (make digit white on black background)
        image = ImageOps.invert(image)
        
        # Apply threshold with a lower value to better detect faint lines
        image = image.point(lambda x: 255 if x > 100 else 0)
        
        # Get the bounding box of the content
        bbox = image.getbbox()
        if bbox:
            # Add padding before cropping
            padding = 10
            bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image.size[0], bbox[2] + padding),
                min(image.size[1], bbox[3] + padding)
            )
            
            # Crop to the content
            image = image.crop(bbox)
            
            # Resize while maintaining aspect ratio
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
    
    def recognize_character(self, image):
        # Preprocess the image
        processed_img = self.preprocess_image(image)
        
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
            
            # Further lower the confidence threshold for better recognition
            if confidence.item() > 0.005:  # Even lower threshold
                # Map prediction to character
                if predicted.item() < 10:
                    return str(predicted.item()), confidence.item()
                elif predicted.item() < 36:
                    # Map to operators and symbols
                    char_idx = predicted.item() - 10
                    if char_idx < len(self.operators):
                        return self.operators[char_idx], confidence.item()
                    else:
                        symbol_idx = char_idx - len(self.operators)
                        return self.symbols[symbol_idx], confidence.item()
        
        return None, 0.0
    
    def parse_expression(self, expression):
        try:
            # Clean up the expression
            expression = expression.replace(' ', '')
            expression = expression.replace('*', ' * ')
            expression = expression.replace('/', ' / ')
            expression = expression.replace('+', ' + ')
            expression = expression.replace('-', ' - ')
            expression = expression.replace('^', ' ** ')
            
            # Parse the expression using SymPy
            expr = parse_expr(expression)
            return expr
        except Exception as e:
            return None
    
    def solve_equation(self, expression):
        try:
            # Parse the expression
            expr = self.parse_expression(expression)
            if expr is None:
                return None, "Invalid expression"
            
            # If it's an equation (contains '=')
            if '=' in expression:
                left, right = expression.split('=')
                left_expr = self.parse_expression(left)
                right_expr = self.parse_expression(right)
                
                if left_expr is None or right_expr is None:
                    return None, "Invalid equation"
                
                # Solve the equation
                solution = sympy.solve(left_expr - right_expr)
                return solution, "Equation solved"
            else:
                # If it's just an expression, evaluate it
                result = expr.evalf()
                return result, "Expression evaluated"
                
        except Exception as e:
            return None, str(e)
    
    def get_step_by_step_solution(self, expression):
        try:
            # Parse the expression
            expr = self.parse_expression(expression)
            if expr is None:
                return None, "Invalid expression"
            
            steps = []
            
            # If it's an equation
            if '=' in expression:
                left, right = expression.split('=')
                left_expr = self.parse_expression(left)
                right_expr = self.parse_expression(right)
                
                if left_expr is None or right_expr is None:
                    return None, "Invalid equation"
                
                # Add steps for solving the equation
                steps.append(f"Original equation: {left} = {right}")
                steps.append(f"Move all terms to left side: {left_expr - right_expr} = 0")
                
                # Solve step by step
                solution = sympy.solve(left_expr - right_expr)
                steps.append(f"Solution: {solution}")
                
            else:
                # If it's an expression, show evaluation steps
                steps.append(f"Original expression: {expression}")
                steps.append(f"Simplified: {expr}")
                steps.append(f"Result: {expr.evalf()}")
            
            return steps, "Success"
            
        except Exception as e:
            return None, str(e)
    
    def split_image_into_characters(self, image):
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply threshold with a lower value
        binary = gray.point(lambda x: 0 if x > 100 else 255)
        
        # Find connected components with a minimum size
        img_array = np.array(binary)
        labeled_array, num_features = ndimage.label(img_array)
        
        # Get bounding boxes for each component
        bboxes = []
        for i in range(1, num_features + 1):
            coords = np.where(labeled_array == i)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Filter out very small components
                if (x_max - x_min) > 5 and (y_max - y_min) > 5:
                    bboxes.append((x_min, y_min, x_max, y_max))
        
        # Sort bounding boxes by x coordinate
        bboxes.sort(key=lambda x: x[0])
        
        # Extract characters with padding
        characters = []
        for bbox in bboxes:
            # Add padding to the bounding box
            padding = 5
            padded_bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image.size[0], bbox[2] + padding),
                min(image.size[1], bbox[3] + padding)
            )
            char_img = image.crop(padded_bbox)
            characters.append(char_img)
        
        return characters 