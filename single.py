from typing import Any
import whisperx  # type: ignore
from utils import get_device_string, convert_to_mp3
import gc
import torch
import json

# Clear GPU cache and collected garbage
gc.collect()
torch.cuda.empty_cache()

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
        batch_size = 8  # Reduced batch size to fit GPU memory
        compute_type = "float16"
        language_code = "ar"

        # Load model
        model = whisperx.load_model("large-v2", self.device, compute_type=compute_type)

        # Load audio
        audio = whisperx.load_audio(input_path)

        # Transcribe audio
        transcribe_result = model.transcribe(audio, batch_size=batch_size)

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code, device=self.device)
        align_result = whisperx.align(
            transcribe_result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        # Load diarization model
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )
        diarize_segments = diarize_model(audio)

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        # Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, align_result)

        # Final memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return result

ID = "Ar1"
AUDIO_FILE = f"./data/silver/{ID}.mp3"
convert_to_mp3(f"./data/bronze/{ID}.m4a", AUDIO_FILE)

service = Service()
result = service.run(AUDIO_FILE)

# Save result to JSON
with open(f"./data/gold/{ID}.json", "w", encoding="utf-8") as f:
    json.dump(result, f)

# Path to JSON file
json_dateipfad = f"./data/gold/{ID}.json"

# Function to load and process JSON data
def verarbeite_json(json_dateipfad):
    # Load JSON data
    with open(json_dateipfad, 'r', encoding='utf-8') as datei:
        daten = json.load(datei)

    # List for formatted text
    formatierter_text = []

    # Iterate through segments
    for segment in daten['segments']:
        speaker = segment.get('speaker', 'SPEAKER_??')  # Default if no speaker
        text = segment.get('text', '???').strip()  # Default if no text
        formatierter_text.append(f'{speaker}: {text}')

    # Format text for output
    gesamttext = '\n'.join(formatierter_text)

    return gesamttext

# Process JSON data and get formatted text
formatierter_text = verarbeite_json(json_dateipfad)

# Save formatted text to a text file
ausgabe_dateipfad = f"./data/gold/{ID}.txt"
with open(ausgabe_dateipfad, 'w', encoding='utf-8') as ausgabe_datei:
    ausgabe_datei.write(formatierter_text)

print(f'Die Daten wurden erfolgreich in {ausgabe_dateipfad} gespeichert.')
