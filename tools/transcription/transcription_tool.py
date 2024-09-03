from typing import Any

import whisperx

BATCH_SIZE = 8
COMPUTE_TYPE = "int8"
LANGUAGE_CODE = "de"
DEVICE = get_device_string()
HF_TOKEN = "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"

model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
model_a, metadata = whisperx.load_align_model(LANGUAGE_CODE, device=DEVICE)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)

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
