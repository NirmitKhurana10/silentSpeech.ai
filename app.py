from flask import Flask, request, jsonify, send_from_directory
import os
from transcriber import transcribe_video  # Make sure your function works correctly

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

latest_video_path = None

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return no content (204) for favicon request

from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')

# Route to upload the video
@app.route('/upload', methods=['POST'])  # Changed GET to POST for file uploads
def upload():
    global latest_video_path
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    video = request.files['video']
    
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the video file
    path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(path)
    latest_video_path = path
    return jsonify({'video_url': f'/uploads/{video.filename}'}), 200

# Route to serve uploaded video
@app.route('/uploads/<path:filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to transcribe video
@app.route('/transcribe', methods=['GET'])
def transcribe():
    if not latest_video_path:
        return jsonify({'transcription': "No video uploaded."}), 400

    # Call the transcribe function from transcriber.py
    transcription = transcribe_video(latest_video_path)
    return jsonify({'transcription': transcription}), 200

if __name__ == '__main__':
    app.run(debug=True)
