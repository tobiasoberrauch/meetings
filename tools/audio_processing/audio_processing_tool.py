import os

from pydub import AudioSegment


def convert_to_mp3(source_file, target_file):
    if not os.path.exists(target_file):
        audio = AudioSegment.from_file(source_file, format="m4a")
        audio.export(target_file, format="mp3")
        print(f"Konvertiert nach {target_file}")
    else:
        print(f"Datei {target_file} existiert bereits.")

def process_audio(uploaded_file, target_format="mp3"):
    # Pfad zum temporären Speichern der hochgeladenen Datei
    saved_path = f"./temp/{uploaded_file.name}"
    
    # Datei speichern
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Wenn die Datei nicht im Zielformat ist, konvertiere sie
    if uploaded_file.type != f"audio/x-{target_format}":
        converted_path = saved_path.rsplit('.', 1)[0] + f".{target_format}"
        convert_to_mp3(saved_path, converted_path)
        return converted_path
    
    return saved_path
