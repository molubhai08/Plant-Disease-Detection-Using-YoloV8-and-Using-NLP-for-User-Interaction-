from ultralytics import YOLO
import speech_recognition as sr
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import google.generativeai as genai
import requests
import io
import pygame
import time

# ElevenLabs TTS Class
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
        """
        Convert text to speech and play it immediately
        
        Args:
            text (str): Text to convert to speech
            voice_id (str): ID of the voice to use
        """
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(
            url,
            headers=self.headers,
            json=data
        )

        if response.status_code == 200:
            audio_stream = io.BytesIO(response.content)
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        else:
            print(f"Error: {response.status_code}")
            print(response.text)

# Initialize ElevenLabs API
tts = ElevenLabsAPI(api_key="sk_7c1ffa791474c88e72fc603779e3ddda21901f6565540a2d")

genai.configure(api_key="AIzaSyDzV9rClBEZaoBE_MDGfN0-VmFrB9D2fSo")

def hinglish(prompt):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat()
    response = chat.send_message(f"Translate the sentence '{prompt}' into Hinglish (Hindi written in Roman script):", stream=True)
    generated_text = ''.join(chunk.text for chunk in response)
    return generated_text

# Load your trained YOLOv8 model
model = YOLO(r'C:\Users\SARTHAK\Desktop\plant_detection\best(1).pt')

# Run inference on an image
results = model(r'C:\Users\SARTHAK\Desktop\plant_detection\0ebea6f4-08e4-4380-86f8-34d854697e32___JR_FrgE.S 2877.JPG')

class_name = None
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        if model.names:
            class_name = model.names[class_id]
            print(f"Class Name: {class_name}")

name = str(class_name)

recognizer = sr.Recognizer()
translator = GoogleTranslator(source='auto', target='en')

def record_voice_input():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            print("Listening for voice input...")
            audio = recognizer.listen(source)
            recognized_text = recognizer.recognize_google(audio)
            return recognized_text
    except sr.RequestError as e:
        return f"Error: Could not request results from Google Speech Recognition service; {e}"
    except sr.UnknownValueError:
        return "Error: Could not understand the audio."

stop_words = set(stopwords.words('english'))

# Disease data dictionaries (keeping your existing dictionaries)


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


ques = ["danger", "dangers", "dangerous", "effect", "effects", "fix", "treat", "treatment"]

answer = disease_data.get(class_name.lower(), None)

if answer:
    ans = input("Any questions? (y/n): ").strip().lower()

    while ans == 'y':
        response = record_voice_input()
        translated_response = translator.translate(response)
        words = word_tokenize(str(translated_response))

        filtered_sentence = [word.lower() for word in words if word.lower() not in stop_words]

        print("Filtered words:", filtered_sentence)

        a = len(ques)
        for i, q in enumerate(ques):
            if q in filtered_sentence:
                a = i
                break

        if a == len(ques):
            hinglish_sent = hinglish(answer['des'])
            tts.text_to_speech_and_play(hinglish_sent)  # Using ElevenLabs instead of pyttsx3
        elif a in [0, 1, 2]:
            hinglish_sent = hinglish(answer['dan'])
            tts.text_to_speech_and_play(hinglish_sent)  # Using ElevenLabs instead of pyttsx3
        elif a in [3, 4]:
            hinglish_sent = hinglish(answer['eff'])
            tts.text_to_speech_and_play(hinglish_sent)  # Using ElevenLabs instead of pyttsx3
        else:
            hinglish_sent = hinglish(answer['Treat'])
            tts.text_to_speech_and_play(hinglish_sent)  # Using ElevenLabs instead of pyttsx3

        ans = input("Any more questions? (y/n): ").strip().lower()
else:
    print("No disease data available for the detected class.")








        




    






