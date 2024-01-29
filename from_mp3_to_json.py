import os
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
import json
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from utils import get_device

def transcribe_file(file_path):
    device = get_device()
    pipe: AutomaticSpeechRecognitionPipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device=device,
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"}
    )

    outputs = pipe(
        file_path,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
        generate_kwargs={"language": "german"}
    )
    return outputs

def process_file(filename, progress):
    audio_path = os.path.join(audio_directory, filename)
    json_file = os.path.join(transcription_directory, f"{os.path.splitext(filename)[0]}.json")

    if os.path.exists(json_file):
        progress.update(task1, advance=1, description=f"Skipping {filename}")
    else:
        progress.update(task1, advance=1, description=f"Transcribing {filename}")
        transcription_results = transcribe_file(audio_path)
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(transcription_results, file, ensure_ascii=False, indent=4)
        progress.console.print(f"Transcription JSON for {audio_path} saved to {json_file}")

audio_directory = 'data/silver'
transcription_directory = 'data/gold'

if not os.path.exists(transcription_directory):
    os.makedirs(transcription_directory)

audio_files = sorted([f for f in os.listdir(audio_directory) if f.endswith('.mp3') or f.endswith('.wav')])

if __name__ == "__main__":
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task1 = progress.add_task("[red]Processing audio files...", total=len(audio_files))
        for file in audio_files:
            process_file(file, progress)
