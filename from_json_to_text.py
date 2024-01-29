import json
import os


def extract_text_from_json(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.json', '.txt'))

            # Check if the output file already exists
            if not os.path.exists(output_path):
                with open(input_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    text = data.get('text', '')  # Extract the 'text' value

                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(text)


input_folder = './data/gold'
output_folder = './data/gold'

# Extract text from JSON files
extract_text_from_json(input_folder, output_folder)
