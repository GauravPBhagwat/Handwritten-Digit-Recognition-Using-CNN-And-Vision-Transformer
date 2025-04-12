import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from vit_mnist import ViT
from streamlit_drawable_canvas import st_canvas
import gtts
import os
import tempfile

class CurrencyReader:
    def __init__(self):
        self.model = ViT(
            image_size=28,
            patch_size=4,
            num_classes=10,  # Only digits for currency
            dim=256,
            depth=8,
            heads=8,
            mlp_dim=512,
            channels=1,
            dim_head=32,
            dropout=0.1,
            emb_dropout=0.1
        )
        # Load the best model
        checkpoint = torch.load('vit_mnist_best.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Available languages
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
        # Keep track of temporary files
        self.temp_files = []

    def preprocess_digit(self, image_array):
        # Convert to PIL Image
        image = Image.fromarray(image_array.astype('uint8'))
        image = image.convert('L')
        
        # Simple thresholding to make it binary
        image = image.point(lambda x: 255 if x > 128 else 0)
        
        # Find bounding box
        bbox = ImageOps.invert(image).getbbox()
        if bbox:
            image = image.crop(bbox)
            
            # Add small padding
            padding = int(min(image.size) * 0.1)
            image = ImageOps.expand(image, border=padding, fill=0)
            
            # Resize while maintaining aspect ratio
            target_size = 20
            ratio = min(target_size / image.size[0], target_size / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Center in 28x28 image
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
            # Generate speech using gTTS
            tts = gtts.gTTS(text=text, lang=language, slow=False)
            
            # Create a temporary file in the current directory
            temp_file = f"temp_audio_{len(self.temp_files)}.mp3"
            tts.save(temp_file)
            self.temp_files.append(temp_file)
            return temp_file
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                st.error(f"Error cleaning up temporary file: {str(e)}")
        self.temp_files = []

def draw_guidelines(image_array, num_digits):
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    segment_width = width // num_digits
    
    # Draw simple, thin lines
    for i in range(1, num_digits):
        x = i * segment_width
        draw.line([(x, 0), (x, height)], 
                 fill=(100, 100, 100, 128),
                 width=1)
    
    return np.array(image)

def main():
    st.set_page_config(layout="wide")
    
    if 'reader' not in st.session_state:
        st.session_state.reader = CurrencyReader()
    if 'start_new' not in st.session_state:
        st.session_state.start_new = True
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    # Clean up temporary files when the app is closed
    if 'audio_file' in st.session_state and st.session_state.audio_file:
        try:
            if os.path.exists(st.session_state.audio_file):
                os.remove(st.session_state.audio_file)
        except Exception as e:
            st.error(f"Error cleaning up audio file: {str(e)}")
    st.session_state.reader.cleanup()

    st.title("AI Currency Reader")

    # CSS styling
    st.markdown("""
        <style>
        .stCanvas {
            display: flex;
            justify-content: center;
        }
        .prediction-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            border: 2px solid #e0e0e0;
        }
        .prediction-text {
            font-size: 32px;
            font-weight: bold;
            color: #0066cc;
        }
        .warning-text {
            color: #ff4b4b;
            font-size: 18px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        num_digits = st.slider("Number of digits", 1, 5, 3)
        
        # Canvas dimensions
        canvas_height = 200
        digit_width = 100
        total_width = digit_width * num_digits

        # Canvas with guidelines
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=canvas_height,
            width=total_width,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.start_new}"
        )

        if canvas_result.image_data is not None:
            image_with_guidelines = draw_guidelines(canvas_result.image_data, num_digits)
            st.image(image_with_guidelines, use_container_width=True)

        # Language selection
        selected_language = st.selectbox(
            "Select language for speech",
            list(st.session_state.reader.languages.keys())
        )

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            predict_button = st.button("Read Amount", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)

        # Output display container
        output_container = st.empty()

        if clear_button:
            st.session_state.start_new = not st.session_state.start_new
            st.session_state.last_prediction = None
            st.session_state.audio_file = None
            output_container.empty()
            st.rerun()

        if predict_button and canvas_result.image_data is not None:
            predictions = []
            confidences = []
            img_array = canvas_result.image_data
            
            for i in range(num_digits):
                start_x = i * digit_width
                end_x = (i + 1) * digit_width
                start_x = max(0, start_x - 5)
                end_x = min(total_width, end_x + 5)
                digit_section = img_array[:, start_x:end_x]
                
                digit, conf = st.session_state.reader.predict_digit(digit_section)
                if digit is not None:
                    predictions.append(digit)
                    confidences.append(conf)
                else:
                    predictions.append("?")
                    confidences.append(0.0)
            
            if predictions:
                result = ''.join(predictions)
                if "?" not in predictions:
                    st.session_state.last_prediction = result
                    
                    # Generate speech with language-specific text
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
                    
                    audio_file = st.session_state.reader.text_to_speech(
                        amount_text,
                        st.session_state.reader.languages[selected_language]
                    )
                    
                    if audio_file:
                        st.session_state.audio_file = audio_file
                        
                        # Display prediction
                        output_container.markdown(f"""
                            <div class="prediction-box">
                                <div class="prediction-text">
                                    Amount: ₹{result}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence scores
                        st.write("Confidence scores:")
                        for idx, conf in enumerate(confidences):
                            st.write(f"Digit {idx+1}: {conf:.2f}")
                        
                        # Play audio
                        st.audio(audio_file)
                else:
                    output_container.markdown(f"""
                        <div class="prediction-box">
                            <div class="warning-text">
                                Partial Recognition: {' '.join(predictions)}
                            </div>
                            <div>Please write digits more clearly</div>
                        </div>
                    """, unsafe_allow_html=True)

        elif st.session_state.last_prediction:
            output_container.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-text">
                        Last Amount: ₹{st.session_state.last_prediction}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.audio_file:
                st.audio(st.session_state.audio_file)

        # Tips section
        with st.expander("Tips for better recognition"):
            st.markdown("""
            - Write each digit clearly in its section
            - Use medium-thick strokes
            - Center the digits
            - Keep digits separate
            - Write in a standard format
            - Make sure digits are well-formed
            """)

if __name__ == "__main__":
    main() 