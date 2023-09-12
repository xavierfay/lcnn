import os
from PIL import Image

def convert_jpg_to_png(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter out all JPG files
    jpg_files = [f for f in files if f.endswith('.jpg') or f.endswith('.jpeg')]

    for jpg_file in jpg_files:
        # Full path to the jpg file
        full_jpg_path = os.path.join(directory_path, jpg_file)

        # Create the PNG filename
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        full_png_path = os.path.join(directory_path, png_file)

        # Convert the JPG to PNG
        img = Image.open(full_jpg_path)
        img.save(full_png_path, "PNG")

        # Remove the original JPG
        os.remove(full_jpg_path)

    print(f"Converted {len(jpg_files)} JPG files to PNG.")

if __name__ == "__main__":
    # Path to the folder containing the JPG files
    directory = "C:\\Users\\xavier\\Documents\\GitHub\\lcnn\\data\\dpid_raw\\images\\"

    convert_jpg_to_png(directory)
