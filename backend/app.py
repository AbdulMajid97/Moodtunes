from flask import Flask, request, jsonify
import librosa
import numpy as np
import pandas as pd
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
import joblib

# Load the model and data when the API starts
model = keras.models.load_model('my_cnn_model.h5')
tr_mean = pd.read_csv('tr_mean.csv').values
tr_std = pd.read_csv('tr_std.csv').values
merged_df = pd.read_csv('merged.csv')

app = Flask(__name__)

# Emotion mapping
emotion_mapping = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

def resize_array(array):
    new_matrix = np.zeros((30,150)) 
    for i in range(30):               
        for j in range(150):          
            try:                                 
                new_matrix[i][j] = array[i][j]
            except IndexError:                   
                pass
    return new_matrix

def query_by_emotion(emotion):
    matched_songs = merged_df[merged_df['emotion'] == emotion]
    return matched_songs

def detect_emotion(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    resized_mfccs = resize_array(mfccs)
    scaled_mfccs = (resized_mfccs - tr_mean) / tr_std
    scaled_mfccs = np.resize(scaled_mfccs, (1, 30, 150, 1))
    prediction = model.predict(scaled_mfccs)
    emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_mapping[emotion_index]
    return predicted_emotion

def generate_playlist(audio_file, emotion):
    print("Detected emotion:", emotion)
    if emotion == "fear":
        desired_emotion = "Peaceful"
    elif emotion == "angry":
        desired_emotion = "Calm"
    elif emotion == "sad":
        desired_emotion = "Sad"
    elif emotion == "happy":
        desired_emotion = "Happy"
    elif emotion == "neutral":
        desired_emotion = "Energetic"
    else:
        desired_emotion = emotion

    matched_songs = query_by_emotion(desired_emotion)
    if matched_songs.empty:
        print("No songs found with the desired emotion.")
        return []

    recommended_songs = matched_songs.sort_values(['mean_arousal', 'mean_valence'], ascending=False)[:10]
    recommended_urls = []
    for index, row in recommended_songs.iterrows():
        recommended_urls.append(row['URL'])
        
    return recommended_urls

@app.route('/detect_emotion', methods=['POST'])
def post_audio_file():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    emotion = detect_emotion(filename)
    playlist_urls = generate_playlist(filename, emotion)
    os.remove(filename)
    return jsonify({'emotion': emotion, 'playlist_urls': playlist_urls})  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
