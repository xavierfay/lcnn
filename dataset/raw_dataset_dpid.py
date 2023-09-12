import os
import json
import random


def extract_lines_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = []
        for line in file.readlines():
            # Split by spaces
            parts = line.strip().split()

            # Check if the line starts with the number '1'
            if parts[0] == '1':
                # Extract the coordinates (excluding the first number)
                coords = list(map(float, parts[1:]))
                lines.append(coords)
    print(lines)
    return lines


def convert_txt_to_json_in_directory(directory):
    data = []

    # Loop through each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            full_path = os.path.join(directory, filename)

            # Extract lines from the .txt file
            lines = extract_lines_from_file(full_path)

            # Convert .txt filename to .png
            image_filename = filename.replace('.txt', '.png')

            # Append to the data list
            data.append({
                "filename": image_filename,
                "lines": lines
            })

    # Shuffle the data randomly
    random.shuffle(data)

    # Split the data into test and validation sets
    split_idx = int(0.2 * len(data))
    test_data = data[:split_idx]
    val_data = data[split_idx:]

    # Write the test and validation sets to separate .json files
    with open(os.path.join(directory, 'valid.json'), 'w') as test_json_file:
        json.dump(test_data, test_json_file, indent=4)

    with open(os.path.join(directory, 'train.json'), 'w') as val_json_file:
        json.dump(val_data, val_json_file, indent=4)


# Example usage:
directory = 'C:\\Users\\xavier\\Documents\\Thesis\\Demo_P&IDs\\image_dpid'
convert_txt_to_json_in_directory(directory)
