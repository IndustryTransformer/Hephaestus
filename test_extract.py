import subprocess
from pathlib import Path


def extract_7z_files(input_dir, output_dir):
    """
    Extract all parts of a multi-part 7z archive using the 7z command-line tool.

    Args:
        input_dir (str or Path): Directory containing the 7z parts
        output_dir (str or Path): Directory to extract files to
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .7z.001 files (first parts of archives)
    first_parts = list(input_dir.glob("*.7z.001"))

    if not first_parts:
        print(f"No .7z.001 files found in {input_dir}")
        return

    for first_part in first_parts:
        print(f"Extracting {first_part}...")

        try:
            # Use 7z command-line tool to extract the multi-part archive
            cmd = ["7z", "x", str(first_part), f"-o{str(output_dir)}"]
            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                print(f"Successfully extracted {first_part} to {output_dir}")
            else:
                print(f"Error extracting {first_part}:")
                print(process.stderr)
        except Exception as e:
            print(f"Error executing 7z command: {e}")


if __name__ == "__main__":
    data_dir = Path("./data/3w_dataset")
    extract_dir = data_dir

    print(f"Extracting 7z archives from {data_dir} to {extract_dir}")
    extract_7z_files(data_dir, extract_dir)

    # Check if extraction was successful
    extracted_files = list(extract_dir.glob("**/*.csv"))
    if extracted_files:
        print(f"Successfully extracted {len(extracted_files)} CSV files")
    else:
        print(
            "No CSV files found after extraction. There might be an issue with the archive."
        )
