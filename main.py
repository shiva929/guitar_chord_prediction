import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import warnings
import pickle

warnings.filterwarnings('ignore')

st.markdown(
    """
    <div style="padding:10px;text-align:center;">
        <marquee style="color:red;font-weight:bold;font-size:20px;">⚠️ Important Announcement: Beta Version - This is not the final product! The website name is not final and is Subject to Change ⚠️</marquee>
    </div>
    """, unsafe_allow_html=True
)

# Add logo
st.image('as12.png', width=150)  # Updated to use st.image for displaying logo

st.title('Guitar Chord Recognition')

st.markdown("<br><br>", unsafe_allow_html=True)

def predict_chords(audio_file, model, label_encoder, segment_length=0.5):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    chords_pred = []
    
    # Loop through the audio in segments
    for start in np.arange(0, duration, segment_length):
        end = min(start + segment_length, duration)
        
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        segment = y[start_frame:end_frame]

        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        features = np.mean(chroma, axis=1).reshape(1, -1)
        
        probas = model.predict_proba(features)[0]
        
        best_idx = np.argmax(probas)
        best_chord = label_encoder.inverse_transform([best_idx])[0]
        chords_pred.append(str(best_chord)) 
    
    return chords_pred

with open('chord_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

audio_file = st.file_uploader('Please upload an audio file (.wav)')

if audio_file is not None:
    st.audio(audio_file)
    if st.button('Predict Chords'):
        predicted_chords = predict_chords(audio_file, model, label_encoder, segment_length=0.5)
        st.write(f'Predicted chords: {predicted_chords}')
