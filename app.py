from typing import Any

import streamlit as st
import whisperx

from utils import (
    convert_to_mp3,
    get_device_string,
)

BATCH_SIZE = 8
COMPUTE_TYPE = "int8"
LANGUAGE_CODE = "de"
DEVICE = get_device_string()
HF_TOKEN = "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"

model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
model_a, metadata = whisperx.load_align_model(LANGUAGE_CODE, device=DEVICE)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)


# Funktion, um das Audio zu transkribieren und die Sprecher zu trennen
def transcribe_audio(input_path: str, device: str) -> Any:
    audio = whisperx.load_audio(input_path)
    transcribe_result = model.transcribe(audio, batch_size=BATCH_SIZE)

    align_result = whisperx.align(
        transcribe_result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    diarize_segments = diarize_model(audio)

    return whisperx.assign_word_speakers(diarize_segments, align_result)


# Streamlit UI
st.title("Audio to Text with Speaker Separation")

uploaded_file = st.file_uploader(
    "Choose an audio file (MP3 or M4A)", type=["mp3", "m4a"]
)

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    # Pfad zum temporären Speichern der hochgeladenen Datei
    saved_path = f"./temp/{uploaded_file.name}"

    # Datei speichern
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Wenn die Datei M4A ist, konvertiere sie zu MP3
    if uploaded_file.type == "audio/x-m4a":
        mp3_path = saved_path.replace(".m4a", ".mp3")
        convert_to_mp3(saved_path, mp3_path)
        input_path = mp3_path
    else:
        input_path = saved_path

    # Audio transkribieren
    result = transcribe_audio(input_path, DEVICE)

    # Ergebnis anzeigen
    st.write(result)
