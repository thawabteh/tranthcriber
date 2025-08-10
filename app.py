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
    languages = request.form.getlist('language')
    accent = request.form.get('accent', 'general')

    final_language = None
    initial_prompt = None
    if 'ar' in languages and 'en' in languages:
        final_language = None
        arabic_prompt_part = ARABIC_ACCENT_PROMPTS.get(accent, "")
        initial_prompt = f"{arabic_prompt_part} This text contains English and Arabic phrases."
    elif 'ar' in languages:
        final_language = 'ar'
        initial_prompt = ARABIC_ACCENT_PROMPTS.get(accent, "")
    elif 'en' in languages:
        final_language = 'en'

    try:
        audio_data = audio_file.read()
        # Use a BytesIO object so we can read the data multiple times
        audio_io = io.BytesIO(audio_data)

        # 1. Get audio duration for estimation
        samples, sr = librosa.load(audio_io, sr=16000, mono=True)
        duration_seconds = librosa.get_duration(y=samples, sr=sr)

        # 2. Transcribe the audio
        segments, _ = model.transcribe(
            samples,
            language=final_language,
            beam_size=5,
            initial_prompt=initial_prompt,
            word_timestamps=True
        )

        # 3. Process segments for SRT and plain text
        srt_content = []
        plain_text_content = []
        subtitle_line_counter = 1

        all_words = []
        for segment in segments:
            plain_text_content.append(segment.text)
            if segment.words:
                all_words.extend(segment.words)

        if all_words:
            current_line = []
            line_start_time = all_words[0].start
            for i, word in enumerate(all_words):
                current_line.append(word.word)
                is_last_word = (i + 1 == len(all_words))
                long_pause_ahead = not is_last_word and (all_words[i + 1].start - word.end > 0.7)
                line_is_long = len(current_line) >= 8

                if is_last_word or long_pause_ahead or line_is_long:
                    line_end_time = word.end
                    start_str = f"{int(line_start_time // 3600):02}:{int(line_start_time % 3600 // 60):02}:{int(line_start_time % 60):02},{int(line_start_time * 1000 % 1000):03}"
                    end_str = f"{int(line_end_time // 3600):02}:{int(line_end_time % 3600 // 60):02}:{int(line_end_time % 60):02},{int(line_end_time * 1000 % 1000):03}"
                    line_text = "".join(current_line).strip()
                    srt_content.append(f"{subtitle_line_counter}\n{start_str} --> {end_str}\n{line_text}\n")
                    subtitle_line_counter += 1
                    current_line = []
                    if not is_last_word:
                        line_start_time = all_words[i + 1].start

    except Exception as e:
        return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 500

    return jsonify({
        "srt_transcription": "\n".join(srt_content),
        "plain_transcript": " ".join(plain_text_content),
        "estimated_processing_time": duration_seconds * 0.15  # Estimate: 15% of audio duration on CPU
    })


