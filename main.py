import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pickle

# Add an important announcement tab with no background and red text
st.markdown(
    """
    <div style="padding:10px;text-align:center;">
        <marquee style="color:red;font-weight:bold;font-size:20px;">⚠️ Important Announcement: Beta Version - This is not the final product! The website name is not final and is Subject to Change ⚠️</marquee>
    </div>
    """, unsafe_allow_html=True
)

# Add logo
st.logo('as12.png',size='large')  # Replace with the path to your logo

# Add some space


st.title('Guitar Chord Recognition')

st.markdown("<br><br>", unsafe_allow_html=True)

def predict_chords(audio_file, model, label_encoder, segment_length=0.5):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    chords_pred = []
    
    # Loop through the audio in segments
    for start in np.arange(0, duration, segment_length):
        end = min(start + segment_length, duration)
        
        # Convert start and end times to frame indices
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        segment = y[start_frame:end_frame]

        # Extract features
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        features = np.mean(chroma, axis=1).reshape(1,-1)
        
        # Get prediction probabilities
        probas = model.predict_proba(features)[0]
        
        # Get all chords with probability above the threshold
        best_idx = np.argmax(probas)
        best_chord = label_encoder.inverse_transform([best_idx])[0]
        chords_pred.append(best_chord)
    
    return chords_pred

# Load the model and label encoder
with open('chord_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Audio file uploader
audio_file = st.file_uploader('Please upload an audio file (.wav)')

if audio_file is not None:
    st.audio(audio_file)
    predicted_chords = predict_chords(audio_file, model, label_encoder, segment_length=0.5)
    if st.button('Predict Chords'):
        st.write(f'Predicted chords: {predicted_chords}')
