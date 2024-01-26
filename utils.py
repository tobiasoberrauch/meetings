import os
import torch
from pydub import AudioSegment

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_device_string():
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def get_device():
    return torch.device(get_device_string())



def convert_to_mp3(source_file, target_file):
    if not os.path.exists(target_file):
        audio = AudioSegment.from_file(source_file, format="m4a")
        audio.export(target_file, format="mp3")
        print(f"Konvertiert nach {target_file}")
    else:
        print(f"Datei {target_file} existiert bereits.")

def convert_to_wav(source_file, target_file):
    if not os.path.exists(target_file):
        audio = AudioSegment.from_file(source_file, format="m4a")
        audio.export(target_file, format="wav")
        print(f"Konvertiert nach {target_file}")
    else:
        print(f"Datei {target_file} existiert bereits.")