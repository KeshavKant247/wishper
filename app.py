import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify
import whisper
import os
import uuid

# Initialize Flask and Whisper model
app = Flask(__name__)
model = whisper.load_model("large-v3")  # Use "tiny", "small", etc. for faster performance

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if not request.json or 'audio_url' not in request.json:
        return jsonify({'error': 'Missing audio_url in request'}), 400

    audio_url = request.json['audio_url']
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(OUTPUT_DIR, audio_filename)

    try:
        # Download audio file
        with urllib.request.urlopen(audio_url) as response:
            with open(audio_path, 'wb') as f:
                f.write(response.read())

        # Transcribe using Whisper
        result = model.transcribe(audio_path,initial_prompt="This is a sales call between an Adda247 sales representative and a student. The conversation will be about government job exam preparation and will be in a mix of Hindi, English, and Hinglish. Key terms to listen for are: Adda247, अड्डा247, government job, सरकारी नौकरी, Bank Maha Pack, SSC CGL Maha Pack, test series, mock test, video course, live classes, validity, double validity, offer price, discount, payment, EMI option, faculty, guidance, study plan. Exam names include SBI PO, IBPS Clerk, RRB NTPC, UPSC, CTET, Prelims, and Mains. Subjects are Quantitative Aptitude, Reasoning, General Awareness, and Current Affairs. Common phrases are: aapko isme kya milega, iski keemat kya hai, course access kab tak hai, doubt session, syllabus, तैयारी, percentile.")
        transcript = result['text']

    except Exception as e:
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500

    finally:
        # Delete the audio file after transcription
        if os.path.exists(audio_path): 
            os.remove(audio_path)

    return jsonify({'transcript': transcript})

if __name__ == '__main__':
    app.run(debug=True, port=5009)

