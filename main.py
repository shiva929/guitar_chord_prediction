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
st.logo('as12.png', size='large')  

st.title('Guitar Chord Recognition')

st.markdown("<br><br>", unsafe_allow_html=True)

def predict_chords(audio_file, model, label_encoder, sr=22050, threshold=0.3):
    y, sr = librosa.load(audio_file, sr=sr)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    st.write(f"Detected BPM: {tempo[0]:.2f}")
    
    chords_pred = []
    
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i + 1]
        
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        segment = y[start_frame:end_frame]

        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        features = np.mean(chroma, axis=1).reshape(1, -1)
        
        probas = model.predict_proba(features)[0]
        
        chord_list = []
        for idx, proba in enumerate(probas):
            if proba > threshold:
                chord = label_encoder.inverse_transform([idx])[0]
                chord_list.append(str(chord))
        
        if chord_list:
            chords_pred.append(chord_list[0])  # Choose the first chord if multiple are found
        else:
            chords_pred.append("Rest")  # Mark as "Rest" if no chord is predicted
    
    return chords_pred

with open('chord_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

audio_file = st.file_uploader('Please upload an audio file (.wav)')

if audio_file is not None:
    st.audio(audio_file)
    if st.button('Predict Chords'):
        predicted_chords = predict_chords(audio_file, model, label_encoder, sr=22050, threshold=0.1)
        st.write(f'Predicted chords: {predicted_chords}')
