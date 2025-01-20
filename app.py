from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import speech_recognition as sr
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pygame
import time
import requests
import io

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ElevenLabsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        pygame.mixer.init()

    def text_to_speech_and_play(self, text, voice_id="zrHiDhphv9ZnVXBqCLjz"):
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }

        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            audio_stream = io.BytesIO(response.content)
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

# Initialize components
tts = ElevenLabsAPI(api_key="sk_ac4225b31fb971c3b6c252de8793cb8092fc24e8d22429c2")
model = YOLO(r'C:\Users\SARTHAK\Desktop\plant_detection\best(1).pt')
recognizer = sr.Recognizer()
translator = GoogleTranslator(source='auto', target='en')
stop_words = set(stopwords.words('english'))

disease_data = {
    "apple black rot": {
        "des": "Ye fungus Botryosphaeria obtusa ke wajah se hota hai. Ye garam aur humid climate me thrive karta hai. Dead wood aur mummified fruits me overwinter kar sakta hai.",
        "dan": "Fruits ko unmarketable banakar significant yield loss karta hai. Tree ki health ko time ke saath weak kar deta hai, productivity reduce karta hai. Wind-borne spores, rain splash, aur infected pruning tools ke through spread hota hai.",
        "eff": "Apple Black Rot zyadatar apple trees aur closely related fruit trees ko target karta hai, lekin unrelated crops ko significantly affect nahi karta.",
        "Treat": "Dormancy ke time infected branches ko prune karke burn ya dispose kar do. Orchard se mummified fruits hatao. Resistant apple varieties use karo."
    },
    "grape black rot": {
        "des": "Ye fungus Guignardia bidwellii ke wajah se hota hai. Garam aur wet climates, especially poorly ventilated vineyards, me thrive karta hai.",
        "dan": "Fruit clusters ko destroy karke severe yield loss karta hai. Vines ko weak karta hai, future production cycles ko impact karta hai.",
        "eff": "Ye sirf grapevines ko affect karta hai; doosre crops ko significantly impact nahi karta.",
        "Treat": "Infected plant parts ko prune karke mummified berries ko remove karo. Vineyards me proper spacing aur ventilation ensure karo. Splash dispersal reduce karne ke liye mulch apply karo."
    },
    "potato early blight": {
        "des": "Ye fungus Alternaria solani ke wajah se hota hai. Garam aur humid conditions me thrive karta hai aur prolonged wet periods ke dauraan sabse severe hota hai.",
        "dan": "Plants ko weak aur tubers ko damage karke significant yield reduction karta hai. Crop ki quality aur storage life ko affect karta hai.",
        "eff": "Ye tomatoes (jo potatoes ke closely related hain) aur nightshade family ke doosre plants ko bhi infect kar sakta hai.",
        "Treat": "Certified disease-free seed potatoes plant karo. Pathogen build-up reduce karne ke liye crop rotation karo. Infected plant debris ko remove aur destroy karo."
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run YOLO model
        results = model(filepath)
        class_name = None
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if model.names:
                    class_name = model.names[class_id]
                    break
            if class_name:
                break

        os.remove(filepath)

        if class_name:
            return jsonify({'success': True, 'disease': class_name})
        else:
            return jsonify({'error': 'No disease detected'})
        
ques = ["danger", "dangers", "dangerous", "effect", "effects", "fix", "treat", "treatment"]

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    question = request.json.get('question', '')  # Fresh question retrieval

    if not question:
        return jsonify({'error': 'No question provided'})
    
    translated = translator.translate(question)  # Translate new question
    
    disease_name = request.json.get('disease', '').lower()  # Retrieve disease dynamically
    name = str(disease_name)
    answer = disease_data.get(name.lower(), None)

    disease_info = disease_data.get(disease_name, {})
    if not disease_info:
        return jsonify({'error': 'Disease information not found'})
    
    # Tokenize and filter words
    word_tokens = word_tokenize(translated.lower())
    filtered_words = [word for word in word_tokens if word not in stop_words]

    a = len(ques)  # Reset index tracker
    for i in range(len(ques)):
        for j in range(len(filtered_words)):
            if filtered_words[j] == ques[i]:
                a = i
            else:
                continue

    # Generate response dynamically
    if a == len(ques):
        response_text = tts.text_to_speech_and_play(answer['des'])  # Using ElevenLabs
    elif a in [5, 6, 7]:
        response_text = tts.text_to_speech_and_play(answer['Treat'])  # Using ElevenLabs
    elif a in [3 ,4]:
        response_text = tts.text_to_speech_and_play(answer['eff'])  # Using ElevenLabs
    else:
        response_text = tts.text_to_speech_and_play(answer['dan'])

    return jsonify({'response': response_text})

@app.route('/voice_question', methods=['POST'])
def voice_question():
    file = request.files['audio']
    if not file:
        return jsonify({'error': 'No audio file provided'})

    audio_data = sr.AudioFile(file)
    with audio_data as source:
        audio = recognizer.record(source)
    try:
        question = recognizer.recognize_google(audio)
        return jsonify({'question': question})
    except sr.UnknownValueError:
        return jsonify({'error': 'Unable to recognize speech'})
    except sr.RequestError as e:
        return jsonify({'error': f'Request error: {e}'})

if __name__ == '__main__':
    app.run(debug=True)

