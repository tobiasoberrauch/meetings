import json

# Pfad zur JSON-Datei
json_dateipfad = 'data.json'

# Funktion zum Laden und Verarbeiten der JSON-Daten
def verarbeite_json(json_dateipfad):
    # JSON-Daten laden
    with open(json_dateipfad, 'r', encoding='utf-8') as datei:
        daten = json.load(datei)

    # Liste für den formatierten Text
    formatierter_text = []

    # Durch die Segmente iterieren
    for segment in daten['segments']:
        speaker = segment.get('speaker', 'SPEAKER_??')  # Standardwert, falls kein Sprecher vorhanden ist
        text = segment.get('text', '???').strip()  # Standardwert, falls kein Text vorhanden ist
        formatierter_text.append(f'{speaker}: {text}')

    # Formatieren Sie den Text für die Ausgabe
    gesamttext = '\n'.join(formatierter_text)

    return gesamttext

# Verarbeite die JSON-Daten und erhalte den formatierten Text
formatierter_text = verarbeite_json(json_dateipfad)

# Speichere den formatierten Text in einer Textdatei
ausgabe_dateipfad = 'transkript.txt'  # Zieltextdatei
with open(ausgabe_dateipfad, 'w', encoding='utf-8') as ausgabe_datei:
    ausgabe_datei.write(formatierter_text)

print(f'Die Daten wurden erfolgreich in {ausgabe_dateipfad} gespeichert.')
