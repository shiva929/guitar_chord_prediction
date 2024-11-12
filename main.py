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

# Add logo
st.image('as12.png', width=200)  # Update to use st.image for displaying the logo

st.title('Guitar Chord Recognition')


# Function to load the model and label encoder
def load_model():
    with open('chord_svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

# Function to predict chords from the audio file
def predict_chords(audio_file, model, label_encoder, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    
    # Detect tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Display detected BPM
    st.metric(label="Detected BPM", value=f"{tempo[0]:.2f}")
    
    chords_pred = []
    
    # Iterate through beat frames to predict chords
    for i in range(len(beat_times)-1):
        start = beat_times[i]
        end = beat_times[i + 1]
        
        # Convert start and end times to frame indices
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        segment = y[start_frame:end_frame]

        # Extract features
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        features = np.mean(chroma, axis=1).reshape(1, -1)
        
        # Get prediction probabilities
        probas = model.predict_proba(features)[0]
        
        # Get the predicted chord based on the highest probability
        predicted_class = np.argmax(probas)
        chord = label_encoder.inverse_transform([predicted_class])[0]
        
        chords_pred.append(str(chord))
    
    return y, sr, beat_times, chords_pred, tempo

# Function to display the waveform with detected beats and predicted chords
def display_chord_waveform(y, sr, beat_times, chords_pred, tempo):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')

    # Annotate the chords on the graph
    for i in range(len(beat_times) - 1):
        plt.text(beat_times[i] + 0.02, 0.5, chords_pred[i], 
                 horizontalalignment='left', fontsize=10, color='blue')
    
    plt.title(f'Waveform with Detected Beats and Predicted Chords (Tempo: {tempo[0]:.2f} BPM)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.ylim([-1, 1])  # Set y-limits for better visibility
    st.pyplot(plt)

# Load the model and label encoder
model, label_encoder = load_model()

# File uploader for audio file
audio_file = st.file_uploader('Please upload an audio file (.wav or .mp3)')

if audio_file is not None:
    # Check if the file type is correct
    if audio_file.type not in ['audio/wav', 'audio/mpeg']:
        st.error('Please upload a valid WAV or MP3 file.')
    else:
        st.audio(audio_file)
        
        if st.button('Predict Chords'):
            with st.spinner('Processing...'):
                y, sr, beat_times, predicted_chords, tempo = predict_chords(audio_file, model, label_encoder, sr=22050)
                st.write(f'Predicted chords: {", ".join(predicted_chords)}')
                display_chord_waveform(y, sr, beat_times, predicted_chords, tempo)
