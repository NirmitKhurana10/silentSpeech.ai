import tensorflow as tf
import numpy as np
import cv2
import os
import requests

# === Step 1: Auto-download model from Google Drive ===
def download_model_from_gdrive():
    file_id = "1iYJ0URxfGm_IcFkGCYUWhDRW71j6vtjb"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    model_path = "silentSpeech_model.h5"

    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully.")

download_model_from_gdrive()

# === Step 2: Load trained model ===
model = tf.keras.models.load_model("silentSpeech_model.h5", compile=False)

# === Step 3: Vocabulary and decoding layers ===
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# === Step 4: Preprocess video ===
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < 75:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (140, 46))
        frames.append(resized)

    cap.release()

    while len(frames) < 75:
        frames.append(np.zeros((46, 140), dtype=np.uint8))

    video = np.array(frames).astype(np.float32) / 255.0
    video = np.expand_dims(video, axis=-1)
    video = np.expand_dims(video, axis=0)

    return video

# === Step 5: Transcribe video ===
def transcribe_video(video_path):
    try:
        input_tensor = preprocess_video(video_path)
        yhat = model.predict(input_tensor)
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=False)[0][0].numpy()
        text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')
        return text
    except Exception as e:
        return f"Error: {str(e)}"
    