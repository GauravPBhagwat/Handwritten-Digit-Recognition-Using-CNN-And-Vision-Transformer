from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import numpy as np
import base64
import io
from math_solver import MathSolver

app = Flask(__name__)
solver = MathSolver()

@app.route('/')
def index():
    return render_template('math_solver.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        img_array = np.array(image)
        
        # Check if image is empty
        if np.all(img_array == 0):
            return jsonify({
                'success': False,
                'message': 'No expression detected. Please draw a mathematical expression.',
                'steps': None
            })
        
        # Split image into characters
        characters = solver.split_image_into_characters(image)
        
        if not characters:
            return jsonify({
                'success': False,
                'message': 'No characters detected. Please write more clearly.',
                'steps': None
            })
        
        # Recognize each character
        expression = ''
        confidences = []
        
        for char_img in characters:
            char, conf = solver.recognize_character(char_img)
            if char is not None:
                expression += char
                confidences.append(conf)
            else:
                expression += '?'
                confidences.append(0.0)
        
        if '?' in expression:
            return jsonify({
                'success': False,
                'message': 'Some characters could not be recognized. Please write more clearly.',
                'expression': expression,
                'steps': None
            })
        
        # Get step-by-step solution
        steps, message = solver.get_step_by_step_solution(expression)
        
        if steps is None:
            return jsonify({
                'success': False,
                'message': message,
                'expression': expression,
                'steps': None
            })
        
        return jsonify({
            'success': True,
            'expression': expression,
            'steps': steps,
            'confidences': confidences
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'steps': None
        })

if __name__ == '__main__':
    app.run(debug=True) 