# # import tensorflow as tf
# # import numpy as np
# # import librosa
# # import subprocess
# # import os

# # # Load the trained model
# # model = tf.keras.models.load_model("silentSpeech_model.h5", compile=False)

# # # Mapping from indices to characters (adjust as per your training)
# # characters = ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
# #               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

# # # Vectorized mapping from numeric indices to characters
# # num_to_char = tf.keras.layers.StringLookup(
# #     vocabulary=characters, 
# #     mask_token=None, 
# #     oov_token="[UNK]", 
# #     invert=True
# # )

# # def extract_audio(video_path, audio_path):
# #     """Extract audio from the video using ffmpeg"""
# #     command = [
# #         'ffmpeg',
# #         '-y',                    # overwrite output file if exists
# #         '-i', video_path,
# #         '-vn',                   # disable video
# #         '-acodec', 'pcm_s16le',  # WAV format
# #         '-ar', '16000',          # 16kHz sample rate
# #         '-ac', '1',              # mono audio
# #         audio_path
# #     ]
# #     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# # def transcribe_audio(video_path):
# #     """Extract audio from video and transcribe it using the trained model"""
    
# #     # Prepare the paths
# #     audio_path = video_path.replace(".mp4", ".wav")
# #     extract_audio(video_path, audio_path)

# #     # Load and preprocess audio
# #     audio, _ = librosa.load(audio_path, sr=16000)
# #     audio = np.pad(audio, (0, max(0, 16000 * 5 - len(audio))))
# #     audio = audio[:16000 * 5]

# #     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=46, fmax=8000)
# #     mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
# #     mel_spectrogram = mel_spectrogram[:, :140]
# #     mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 140 - mel_spectrogram.shape[1])))

# #     mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
# #     mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
# #     mel_spectrogram = np.resize(mel_spectrogram, (1, 75, 46, 140, 1))

# #     # Predict using the model
# #     yhat = model.predict(mel_spectrogram)

# #     # Decode using CTC decoding
# #     decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=False)[0][0].numpy()
# #     pred_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')

# #     return pred_text



# import tensorflow as tf
# import numpy as np
# import librosa
# import subprocess
# import os

# # Load the trained model
# model = tf.keras.models.load_model("silentSpeech_model.h5", compile=False)

# # Define vocabulary and create mapping layers
# vocab = ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
#          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# num_to_char = tf.keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# )

# def extract_audio(video_path, audio_path):
#     """Extract audio from the video using ffmpeg"""
#     command = [
#         'ffmpeg',
#         '-y',                    # overwrite output file if exists
#         '-i', video_path,
#         '-vn',                   # disable video
#         '-acodec', 'pcm_s16le',  # WAV format
#         '-ar', '16000',          # 16kHz sample rate
#         '-ac', '1',              # mono audio
#         audio_path
#     ]
#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# def transcribe_audio(video_path):
#     """Extract audio from video and transcribe it using the trained model"""
    
#     # Prepare audio path
#     audio_path = video_path.replace(".mp4", ".wav")
#     extract_audio(video_path, audio_path)

#     # Load and preprocess audio
#     audio, _ = librosa.load(audio_path, sr=16000)
#     audio = np.pad(audio, (0, max(0, 16000 * 5 - len(audio))))
#     audio = audio[:16000 * 5]

#     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=46, fmax=8000)
#     mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     mel_spectrogram = mel_spectrogram[:, :140]
#     mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 140 - mel_spectrogram.shape[1])))

#     mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
#     mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
#     mel_spectrogram = np.resize(mel_spectrogram, (1, 75, 46, 140, 1))

#     # Predict using the model
#     yhat = model.predict(mel_spectrogram)

#     # Decode using CTC decoding and mapping
#     decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=False)[0][0]
#     char_seq = num_to_char(decoded)
#     filtered_chars = tf.ragged.map_flat_values(lambda x: tf.where(x == '', '', x), char_seq)
#     pred_text = tf.strings.reduce_join(filtered_chars[0]).numpy().decode('utf-8')

#     return pred_text


# import tensorflow as tf
# import numpy as np
# import librosa
# import subprocess
# import os

# # Load the trained model
# model = tf.keras.models.load_model("silentSpeech_model.h5", compile=False)

# # Updated vocabulary
# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# # Mapping layers
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# num_to_char = tf.keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# )

# def extract_audio(video_path, audio_path):
#     """Extract audio from the video using ffmpeg"""
#     command = [
#         'ffmpeg',
#         '-y',
#         '-i', video_path,
#         '-vn',
#         '-acodec', 'pcm_s16le',
#         '-ar', '16000',
#         '-ac', '1',
#         audio_path
#     ]
#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# def transcribe_audio(video_path):
#     """Extract audio from video and transcribe it using the trained model"""
    
#     audio_path = video_path.replace(".mp4", ".wav")
#     extract_audio(video_path, audio_path)

#     # Load and preprocess audio
#     audio, _ = librosa.load(audio_path, sr=16000)
#     audio = np.pad(audio, (0, max(0, 16000 * 5 - len(audio))))
#     audio = audio[:16000 * 5]

#     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=46, fmax=8000)
#     mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     mel_spectrogram = mel_spectrogram[:, :140]
#     mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 140 - mel_spectrogram.shape[1])))

#     mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
#     mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
#     mel_spectrogram = np.resize(mel_spectrogram, (1, 75, 46, 140, 1))

#     # Model prediction
#     yhat = model.predict(mel_spectrogram)

#     # Decode and clean up output
#     decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=False)[0][0]
#     char_seq = num_to_char(decoded)
#     filtered_chars = tf.ragged.map_flat_values(lambda x: tf.where(x == '', '', x), char_seq)
#     pred_text = tf.strings.reduce_join(filtered_chars[0]).numpy().decode('utf-8')

#     return pred_text

import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("silentSpeech_model.h5", compile=False)

# Vocabulary and decoding layers
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def preprocess_video(video_path):
    """
    Extract and preprocess 75 grayscale frames from the input video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < 75:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and resize to (140, 46)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (140, 46))  # (W, H)
        frames.append(resized)

    cap.release()

    # Pad if < 75 frames
    while len(frames) < 75:
        frames.append(np.zeros((46, 140), dtype=np.uint8))

    # Convert to float, normalize, and reshape to (1, 75, 46, 140, 1)
    video = np.array(frames).astype(np.float32) / 255.0
    video = np.expand_dims(video, axis=-1)
    video = np.expand_dims(video, axis=0)

    return video  # shape: (1, 75, 46, 140, 1)

def transcribe_video(video_path):
    """
    Preprocess the video and run inference to generate transcription.
    """
    try:
        input_tensor = preprocess_video(video_path)
        yhat = model.predict(input_tensor)

        # Decode using CTC
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=False)[0][0].numpy()

        # Convert from integers to characters
        text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')
        return text
    except Exception as e:
        return f"Error: {str(e)}"