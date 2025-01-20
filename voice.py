import requests
import io
import pygame
import time

class ElevenLabsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        # Initialize pygame mixer
        pygame.mixer.init()

    def get_voices(self):
        """Get available voices"""
        response = requests.get(
            f"{self.base_url}/voices",
            headers=self.headers
        )
        return response.json()

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
            # Create a byte stream from the response content
            audio_stream = io.BytesIO(response.content)
            
            # Load the audio stream into pygame
            pygame.mixer.music.load(audio_stream)
            
            # Play the audio
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print("Audio playback completed")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

# Example usage
if __name__ == "__main__":
    api_key = "sk_7c1ffa791474c88e72fc603779e3ddda21901f6565540a2d"  # Replace with your actual API key
    tts = ElevenLabsAPI(api_key)
    
    # Convert text to speech and play it
    text = "Ye fungus Alternaria solani ke wajah se hota hai. Garam aur humid conditions me thrive karta hai aur prolonged wet periods ke dauraan sabse severe hota hai."
    tts.text_to_speech_and_play(text)

# # Get a Voice object by name or UUID
# voice = eleven.voices["Arnold"]

# # Generate the TTS
# audio = voice.generate("Hey buddy! It's a beautiful day.")



# Set the API key using the set_api_key method (if required)
# elevenlabs.set_api_key('sk_7c1ffa791474c88e72fc603779e3ddda21901f6565540a2d')




