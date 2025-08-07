import os
import io
from flask import Flask, request, jsonify, render_template
from faster_whisper import WhisperModel
import librosa
import numpy as np

# --- App Initialization ---
app = Flask(__name__)

# --- Model Loading ---
print("Loading Whisper model 'large-v3' on CPU...")
# In production, you might want to switch back to GPU if your server has one.
# For now, we'll stick with CPU for compatibility.
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print("Model loaded successfully.")

# --- Accent Prompts Dictionary ---
ARABIC_ACCENT_PROMPTS = {
    "egyptian": "ازيك عامل ايه؟ يا رب تكون بخير. أنا بتكلم باللهجة المصرية.",
    "jordanian": "يا زلمة كيفك شو الأخبار؟ انشالله تمام. هاي هي اللهجة الأردنية.",
    "saudi": "وشلونك؟ عساك طيب. هذي هي اللهجة السعودية.",
    "gulf": "شخبارك؟ عساك بخير. هذي لهجة أهل الخليج.",
    "levantine": "كيفك؟ شو الأخبار؟ ان شاء الله منيح. هاي اللهجة الشامية.",
    "maghrebi": "واش راك؟ لاباس؟ هادي هي اللهجة المغاربية.",
    "iraqi": "شلونك عيني؟ ان شاء الله بخير. هاي اللهجة العراقية.",
    "yemeni": "كيف حالك؟ ان شاء الله طيب. هذه هي اللهجة اليمنية.",
    "general": ""
}


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe_audio_api():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    language = request.form.get('language', 'en')
    accent = request.form.get('accent', 'general')

    initial_prompt = None
    if language == 'ar':
        initial_prompt = ARABIC_ACCENT_PROMPTS.get(accent, "")

    try:
        audio_data = audio_file.read()
        samples, _ = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)

        segments, _ = model.transcribe(
            samples,
            language=language,
            beam_size=5,
            initial_prompt=initial_prompt
        )

        srt_content = []
        for i, segment in enumerate(segments):
            start_time = segment.start
            end_time = segment.end
            start = f"{int(start_time // 3600):02}:{int(start_time % 3600 // 60):02}:{int(start_time % 60):02},{int(start_time * 1000 % 1000):03}"
            end = f"{int(end_time // 3600):02}:{int(end_time % 3600 // 60):02}:{int(end_time % 60):02},{int(end_time * 1000 % 1000):03}"
            srt_content.append(f"{i + 1}\n{start} --> {end}\n{segment.text.strip()}\n")

    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 500

    return jsonify({"srt_transcription": "\n".join(srt_content)})

# The if __name__ == '__main__': block has been removed.
# Gunicorn will run the 'app' object directly.