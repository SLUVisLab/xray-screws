import argparse
import pydicom
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Progress bar
import os

def find_dicom_files(input_path: Path, depth: int = 3):
    """ 
    Finds all files that are exactly `depth` levels below the input_path. 
    This works for cases where DICOM files do not have extensions.
    
    Args:
        input_path (Path): The root directory to start searching from.
        depth (int): How many levels deep to go before collecting files.

    Returns:
        list[Path]: A list of DICOM file paths.
    """
    if depth == 0:
        return [f for f in input_path.iterdir() if f.is_file()]  # Collect files at this depth

    dicom_files = []
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            dicom_files.extend(find_dicom_files(subdir, depth - 1))  # Recursively go deeper

    return dicom_files


def get_patient_folder(dicom_path: Path, levels_up=2):
    """
    Extracts the patient folder name from the DICOM path.
    
    Args:
        dicom_path (Path): Full path to the DICOM file.
        levels_up (int): How many levels up to go to get the patient folder.
    
    Returns:
        str: The patient folder name (e.g., "32423453").
    """
    return dicom_path.parents[levels_up].name

def convert_dicom_to_jpeg(dicom_path: Path, output_base_dir: Path):
    """ Converts a single DICOM file to JPEG and saves it in a patient-specific folder. """
    try:
        dataset = pydicom.dcmread(dicom_path, force=True)

        # Ensure the DICOM file contains pixel data
        if not hasattr(dataset, 'PixelData'):
            return f"Skipping {dicom_path} - No Pixel Data Found"

        # Extract the patient folder name
        patient_folder = get_patient_folder(dicom_path)

        # Create the patient's output directory
        patient_output_dir = output_base_dir / patient_folder
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        # Get next available filename
        existing_files = sorted(patient_output_dir.glob("*.jpg"))
        new_filename = patient_output_dir / f"{len(existing_files):02d}.jpg"

        # Normalize pixel values
        pixel_array = dataset.pixel_array
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255
        pixel_array = pixel_array.astype(np.uint8)

        # Convert to image and save
        img = Image.fromarray(pixel_array)
        img.save(new_filename, quality=95)

        return f"✔ Saved: {new_filename}"

    except Exception as e:
        return f"❌ Error processing {dicom_path}: {e}"

def convert_dicom(input_dir, output_dir, num_threads=4):
    """
    Converts all DICOM files in a directory to JPEGs using multi-threading.
    
    Args:
        input_dir (str or Path): Path to the directory containing DICOM files.
        output_dir (str or Path): Path to the directory to save JPEGs.
        num_threads (int): Number of threads to use for parallel processing.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    dicom_files = find_dicom_files(input_path)
    if not dicom_files:
        print("❌ No DICOM files found.")
        return

    print(f"🚀 Starting DICOM to JPEG conversion with {num_threads} threads...")
    print(f"📂 Found {len(dicom_files)} DICOM files.")
    print(f"📂 Input Directory: {input_path}")
    print(f"💾 Output Directory: {output_path}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(
            executor.map(lambda d: convert_dicom_to_jpeg(d, output_path), dicom_files),
            total=len(dicom_files)
        ))

    # Print summary
    for res in results:
        if res:  # Only print non-empty messages
            print(res)

    print("✅ Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM files to JPEG.")
    parser.add_argument("--input-dir", required=True, help="Absolute path to the input DICOM directory")
    parser.add_argument("--output-dir", required=True, help="Absolute path to the output directory for JPEGs")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)")

    args = parser.parse_args()
    convert_dicom(args.input_dir, args.output_dir, args.threads)