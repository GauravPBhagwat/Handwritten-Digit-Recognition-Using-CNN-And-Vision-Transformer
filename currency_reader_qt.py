import sys
import os
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QSlider, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from vit_mnist import ViT
import gtts
from playsound import playsound

class DrawingCanvas(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: black;
                border: 2px solid #666;
                border-radius: 10px;
            }
        """)
        
        self.drawing = False
        self.last_point = QPoint()
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.digits = []
        self.num_digits = 3
        self.digit_width = self.width() // self.num_digits
        
        # Draw guidelines
        self.draw_guidelines()
        
    def draw_guidelines(self):
        painter = QPainter(self.image)
        painter.setPen(QPen(QColor(100, 100, 100, 128), 1))
        
        for i in range(1, self.num_digits):
            x = i * self.digit_width
            painter.drawLine(x, 0, x, self.height())
        
        painter.end()
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            painter.end()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            
    def clear(self):
        self.image.fill(Qt.black)
        self.draw_guidelines()
        self.update()
        
    def get_image_array(self):
        # Convert QImage to numpy array
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr[:, :, 0]  # Return only the red channel

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
        self.temp_files = []

    def preprocess_digit(self, image_array):
        image = Image.fromarray(image_array.astype('uint8'))
        image = image.convert('L')
        image = image.point(lambda x: 255 if x > 128 else 0)
        
        bbox = ImageOps.invert(image).getbbox()
        if bbox:
            image = image.crop(bbox)
            padding = int(min(image.size) * 0.1)
            image = ImageOps.expand(image, border=padding, fill=0)
            
            target_size = 20
            ratio = min(target_size / image.size[0], target_size / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            final_image = Image.new('L', (28, 28), 0)
            paste_x = (28 - image.size[0]) // 2
            paste_y = (28 - image.size[1]) // 2
            final_image.paste(image, (paste_x, paste_y))
            
            return final_image
        return Image.new('L', (28, 28), 0)

    def predict_digit(self, image_array):
        processed_img = self.preprocess_digit(image_array)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(processed_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() > 0.1:
                return str(predicted.item()), confidence.item()
            return None, 0.0

    def text_to_speech(self, text, language):
        try:
            tts = gtts.gTTS(text=text, lang=language, slow=False)
            temp_file = f"temp_audio_{len(self.temp_files)}.mp3"
            tts.save(temp_file)
            self.temp_files.append(temp_file)
            return temp_file
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error generating speech: {str(e)}")
            return None

    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                QMessageBox.warning(None, "Warning", f"Error cleaning up file: {str(e)}")
        self.temp_files = []

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.reader = CurrencyReader()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('AI Currency Reader')
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f6;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #ddd;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel('AI Currency Reader')
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin: 20px;
            text-align: center;
        """)
        layout.addWidget(title)
        
        # Drawing canvas
        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Number of digits slider
        digits_layout = QVBoxLayout()
        digits_label = QLabel('Number of digits:')
        self.digits_slider = QSlider(Qt.Horizontal)
        self.digits_slider.setMinimum(1)
        self.digits_slider.setMaximum(5)
        self.digits_slider.setValue(3)
        self.digits_slider.valueChanged.connect(self.update_digits)
        digits_layout.addWidget(digits_label)
        digits_layout.addWidget(self.digits_slider)
        controls_layout.addLayout(digits_layout)
        
        # Language selection
        language_layout = QVBoxLayout()
        language_label = QLabel('Select language:')
        self.language_combo = QComboBox()
        self.language_combo.addItems(self.reader.languages.keys())
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        controls_layout.addLayout(language_layout)
        
        layout.addLayout(controls_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.read_button = QPushButton('Read Amount')
        self.read_button.clicked.connect(self.read_amount)
        button_layout.addWidget(self.read_button)
        
        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # Result display
        self.result_label = QLabel('')
        self.result_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin: 20px;
        """)
        layout.addWidget(self.result_label)
        
        # Confidence scores
        self.confidence_label = QLabel('')
        self.confidence_label.setStyleSheet("""
            font-size: 14px;
            color: #666;
            text-align: center;
            margin: 10px;
        """)
        layout.addWidget(self.confidence_label)
        
    def update_digits(self, value):
        self.canvas.num_digits = value
        self.canvas.digit_width = self.canvas.width() // value
        self.canvas.clear()
        
    def clear_canvas(self):
        self.canvas.clear()
        self.result_label.setText('')
        self.confidence_label.setText('')
        
    def read_amount(self):
        if not self.canvas.image.isNull():
            predictions = []
            confidences = []
            img_array = self.canvas.get_image_array()
            
            for i in range(self.canvas.num_digits):
                start_x = i * self.canvas.digit_width
                end_x = (i + 1) * self.canvas.digit_width
                start_x = max(0, start_x - 5)
                end_x = min(self.canvas.width(), end_x + 5)
                digit_section = img_array[:, start_x:end_x]
                
                digit, conf = self.reader.predict_digit(digit_section)
                if digit is not None:
                    predictions.append(digit)
                    confidences.append(conf)
                else:
                    predictions.append("?")
                    confidences.append(0.0)
            
            if predictions:
                result = ''.join(predictions)
                if "?" not in predictions:
                    self.result_label.setText(f'Amount: ₹{result}')
                    
                    # Generate speech
                    selected_language = self.language_combo.currentText()
                    if selected_language == 'English':
                        amount_text = f"The amount is {result} rupees"
                    elif selected_language == 'Hindi':
                        amount_text = f"राशि {result} रुपये है"
                    elif selected_language == 'Marathi':
                        amount_text = f"रक्कम {result} रुपये आहे"
                    elif selected_language == 'Spanish':
                        amount_text = f"La cantidad es {result} rupias"
                    elif selected_language == 'French':
                        amount_text = f"Le montant est de {result} roupies"
                    elif selected_language == 'German':
                        amount_text = f"Der Betrag ist {result} Rupien"
                    elif selected_language == 'Japanese':
                        amount_text = f"金額は{result}ルピーです"
                    elif selected_language == 'Chinese':
                        amount_text = f"金额是{result}卢比"
                    
                    audio_file = self.reader.text_to_speech(
                        amount_text,
                        self.reader.languages[selected_language]
                    )
                    
                    if audio_file:
                        try:
                            playsound(audio_file)
                        except Exception as e:
                            QMessageBox.warning(None, "Warning", f"Error playing audio: {str(e)}")
                    
                    # Display confidence scores
                    conf_text = "Confidence scores:\n"
                    for idx, conf in enumerate(confidences):
                        conf_text += f"Digit {idx+1}: {conf:.2f}\n"
                    self.confidence_label.setText(conf_text)
                else:
                    self.result_label.setText(f'Partial Recognition: {" ".join(predictions)}')
                    self.result_label.setStyleSheet("""
                        font-size: 24px;
                        font-weight: bold;
                        color: #ff4b4b;
                        text-align: center;
                        margin: 20px;
                    """)
                    QMessageBox.warning(None, "Warning", "Please write digits more clearly")
        
    def closeEvent(self, event):
        self.reader.cleanup()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 