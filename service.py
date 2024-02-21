"""The service converts audio file to text with speaker separation"""

import json
from typing import Any

import whisperx  # type: ignore

from utils import get_device_string


class Service:
    """
    Service converts audio to text
    """

    def __init__(self) -> None:
        self.device = get_device_string()
        self.hf_token = "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"

    def run(self, input_path: str) -> Any:
        """
        It runs the service
        """
        batch_size = 16  # reduce if low on GPU mem
        compute_type = "float16"
        language_code = "de"

        model = whisperx.load_model("large-v2", self.device, compute_type=compute_type)
        audio = whisperx.load_audio(input_path)
        transcribe_result = model.transcribe(audio, batch_size=batch_size)

        model_a, metadata = whisperx.load_align_model(language_code, device=self.device)
        align_result = whisperx.align(
            transcribe_result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )
        diarize_segments = diarize_model(audio)

        return whisperx.assign_word_speakers(diarize_segments, align_result)


AUDIO_FILE = "./data/silver/codegaia.mp3"

service = Service()
result = service.run(AUDIO_FILE)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(result, f)
