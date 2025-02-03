import streamlit as st
import librosa
import sounddevice as sd
import joblib
import numpy as np
import tempfile
from warnings import filterwarnings
import smtplib
import speech_recognition as sr
import re
import geocoder
import time
import soundfile as sf

from twilio.rest import Client 

filterwarnings("ignore")

# Constants and preloaded models
RATE = 44100
CHUNK_DURATION = 1.0
CHUNK_SIZE = int(RATE * CHUNK_DURATION)
model = joblib.load('model/mlp_svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Twilio credentials
TWILIO_ACCOUNT_SID = "XXXXXXXXXXXXXXXXXXXXX"  #Enter your twilio account sid
TWILIO_AUTH_TOKEN = "XXXXXXXXXXXXXXXXXXXXXX" #Enter your twilio account auth token
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox Number for whatsapp
TWILIO_PHONE_NUMBER = "+16204551616" # Twilio phone number for sms

# Distress words for detection
distress_words_list = [
    "help", "emergency", "save me", "please", "someone", "now", 
    "danger", "hurt", "pain", "struggling", "need assistance", 
    "call 911", "rescue", "threat", "unsafe", "attack", "fear"
]

def check_distress(transcription):
    transcription_lower = transcription.lower()
    detected_words = [
        word for word in distress_words_list if re.search(r'\b' + re.escape(word) + r'\b', transcription_lower)
    ]
    return detected_words
    
def send_sms(to_number, distress_words=None):
    """
    Sends an SMS alert via Twilio.
    """
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # Play around here with message content
    message_body = (
        f"🚨 Alert! Distress words detected: {', '.join(distress_words)}."
        if distress_words else
        "🚨 Alert! Unusual audio detected."
    )

    try:
        # Send the SMS
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        return True
    except Exception as e:
        print(e)
        return False
    
def send_whatsapp_message(to_number, distress_words=None):
    """
    Sends a WhatsApp alert via Twilio.
    """
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # Construct the message
    message_body = (
        f"🚨 Alert! Distress words detected: {', '.join(distress_words)}."
        if distress_words else
        "🚨 Alert! Unusual audio detected."
    )

    try:
        # Send the WhatsApp message
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f"whatsapp:{to_number}"
        )
        
        return True
    except Exception as e:
        print(e)
        return False

def send_mail(sender="Mailtrap Test <test@demomailtrap.com>", receiver="veda.kosuri@gmail.com", distress_words=None):
    def extract_email(address):
        if "<" in address and ">" in address:
            return address.split("<")[1].strip(">")
        return address

    sender_email = extract_email(sender)
    receiver_email = extract_email(receiver)
    location = get_location()

    if distress_words:
        message = f"""\
Subject: High Level Scream Alert!!!
To: {receiver}
From: {sender}

Scream detected with distress words {distress_words} at {location}!!!""".encode("utf-8")
    else:
        message = f"""\
Subject: Medium Level Scream Alert!!!
To: {receiver}
From: {sender}

Scream detected at {location}!!!""".encode("utf-8")

    try:
        with smtplib.SMTP("live.smtp.mailtrap.io", 587) as server:
            server.starttls()
            server.login("api", "1587e9c98ae13df3350aa1b2f7c75aa8")
            server.sendmail(sender_email, receiver_email, message)
        return True
    except Exception as e:
        print(e)
        return False

def record_audio(duration=10, rate=44100):
    st.write(f"Recording audio for {duration} seconds...")
    audio_data = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    return audio_data.flatten()

def record_audio_intervals():
    intervals = []
    for i in range(3):
        st.write(f"Recording Interval {i + 1} of 3...")
        audio_data = record_audio(duration=10)
        intervals.append(audio_data)
        if i < 2:
            st.write("Pausing for 2 seconds...")
            time.sleep(2)
    return np.concatenate(intervals)

def extract_single_audio_features(y):
    mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=RATE)
    chroma = librosa.feature.chroma_stft(y=y, sr=RATE)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=RATE)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=RATE)
    rmse = librosa.feature.rms(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=RATE)
    onset_env = librosa.onset.onset_strength(y=y, sr=RATE)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=RATE)

    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zero_crossing_rate),
        np.mean(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(rmse),
        np.mean(spectral_rolloff),
        tempo
    ])
    return feature_vector

def process_audio_file(audio_path):
    audio_data, sr = librosa.load(audio_path, sr=RATE)
    st.write(f"Sample rate: {sr}, Duration: {librosa.get_duration(y=audio_data, sr=sr)} seconds")
    num_chunks = len(audio_data) // CHUNK_SIZE
    predictions = []
    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        audio_chunk = audio_data[start:end]
        features = extract_single_audio_features(audio_chunk)
        scaled_features = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled_features)
        predictions.append(prediction[0])
    return predictions

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

def get_location():
    location = geocoder.ip("me")
    return location.city or location.address if location else "Couldn't get location"

def main():
    st.title("Audio Analysis for Surveillance")
    st.write("Record your audio by clicking the button below.")
    mail_sent = False
    whatsapp_message_sent = False
    sms_sent = False

    if st.button("Start Recording"):
        combined_audio = record_audio_intervals()

        if combined_audio is not None:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                sf.write(temp_audio_file.name, combined_audio, RATE)
                temp_audio_file_path = temp_audio_file.name

            predictions = process_audio_file(temp_audio_file_path)
            scream_count = sum(1 for pred in predictions if pred == 1)
            st.write(f"Total screams detected: {scream_count}")

            transcription = transcribe_audio(temp_audio_file_path)
            st.text_area("Audio Transcript", transcription)

            distress_words = check_distress(transcription)
            if scream_count > 0:
                mail_sent = send_mail(distress_words=distress_words if distress_words else None)
                whatsapp_message_sent = send_whatsapp_message("+918499809696", distress_words=distrss_words if distress_words else None) #replace xx with your number
                sms_sent = send_sms("+918499809696", distress_words=distrss_words if distress_words else None)
                
            if mail_sent:
                st.success("Email Alert Sent.")
            
            if whatsapp_message_sent:
                st.success("WhatsApp Alert Sent.")
            
            if sms_sent: 
                st.success("SMS Alert Sent.")
