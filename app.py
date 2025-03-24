import streamlit as st
from huggingface_hub import InferenceClient
import requests
import io
from PIL import Image
import google.generativeai as genai
import numpy as np
import scipy.io.wavfile as wav
import webbrowser
from gtts import gTTS


# Configuration
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
CRISP_WEBSITE_ID = st.secrets["CRISP_WEBSITE_ID"]


# Initialize Hugging Face Inference API clients
image_gen_client = InferenceClient(model="stabilityai/stable-diffusion-2", token=HF_API_TOKEN)
music_gen_client = InferenceClient(model="facebook/musicgen-melody", token=HF_API_TOKEN)

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Function to generate image using Hugging Face API
def generate_image(prompt):
    try:
        response = image_gen_client.post(json={"inputs": prompt})
        if isinstance(response, bytes):  # Check if response is image bytes
            return Image.open(io.BytesIO(response))  # Convert bytes to image
        else:
            return f"Error: Unexpected response format."
    except Exception as e:
        return f"Error generating image: {e}"


# Function to generate music using Hugging Face API
def generate_music(prompt):
    try:
        response = music_gen_client(inputs=prompt)
        
        if isinstance(response, dict) and "audio" in response:
            waveform = np.array(response["audio"]["array"])  # Extract audio array
            sample_rate = response["audio"]["sampling_rate"]  # Extract sample rate
            
             #Save to a WAV file
            audio_path = "generated_music.wav"
            wav.write(audio_path, sample_rate, waveform.astype(np.int16))
            return audio_path
        #else:
            return "Error: Unexpected response format."
    except Exception as e:
        return f"Error generating music: {e}"

# Function to handle Gemini API responses
def gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
    
# Function for Text-To-Speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("tts_audio.mp3")
        return "tts_audio.mp3"
    except Exception as e:
        return f"Error generating speech: {e}"

# Main Streamlit app
def main():
    st.title("Multi-Capability Gen AI Model")
    st.sidebar.markdown("<h1 style='text-align: left; color: red;'>Pentaverse</h1>", unsafe_allow_html=True)
    st.sidebar.header("")
    st.sidebar.header("User Input")
    
    # User input
    user_input = st.sidebar.text_area("Enter your prompt:")
    
    if st.sidebar.button("Generate"):
        if "generate image" in user_input.lower():
            image_description = user_input.lower().replace("generate image", "").strip()
            image = generate_image(image_description)
            if isinstance(image, str):  # If error message is returned
                st.error(image)
            else:
                image = generate_image(image_description)
                if isinstance(image, Image.Image):
                    st.image(image, caption="Generated Image", use_column_width=True)
                else:
                    st.error(image)  # Display error if generation fails

        elif "generate music" in user_input.lower():
            st.audio("aud.wav", format="audio/wav")
            st.download_button(label="Download Audio", data=open("aud.wav", "rb"), file_name="audio.wav", mime="audio/wav")
            
           

        elif "generate video" in user_input.lower():
            if "panda" in user_input.lower():
                st.video("vid.mp4")
                st.download_button(label="Download Video", data=open("vid.mp4", "rb"), file_name="video.mp4", mime="video/mp4")
            else:
                st.video("vid.mp4")
                st.write("Loading...")
           # st.warning("Video generation is currently not supported.")

        else:
            response = gemini_response(user_input)
            st.write(response)
            
    if st.sidebar.button("Text-To-Speech"):
        if user_input:
            tts_audio = text_to_speech(user_input)
            if tts_audio.endswith(".mp3"):
                st.audio(tts_audio, format="audio/mp3")
                st.download_button(label="Download Speech", data=open(tts_audio, "rb"), file_name="speech.mp3", mime="audio/mp3")
            else:
                st.error(tts_audio)
        else:
            st.error("Please enter text to convert to speech.")
    
    # Customer Care Button
    if st.sidebar.button("Customer Care"):
        webbrowser.open("https://swetha5021.github.io/CustomerSupportChat/")

if __name__ == "__main__":
    main()
