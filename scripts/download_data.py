"""Download and cache CFPB Consumer Complaints dataset."""

import os
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
BACKUP_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/"
CSV_PATH = DATA_DIR / "complaints.csv"
ZIP_PATH = DATA_DIR / "complaints.csv.zip"


def download_cfpb(force: bool = False) -> Path:
    """Download CFPB complaints CSV if not already cached.

    Returns path to the unzipped CSV file.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if CSV_PATH.exists() and not force:
        print(f"Data already exists at {CSV_PATH}")
        return CSV_PATH

    print(f"Downloading CFPB data from {CSV_URL}...")
    response = requests.get(CSV_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    ZIP_PATH.unlink()

    if not CSV_PATH.exists():
        # Archive layout may have changed — look for any CSV
        csvs = list(DATA_DIR.glob("*.csv"))
        if csvs:
            csvs[0].rename(CSV_PATH)
            print(f"Renamed {csvs[0].name} -> {CSV_PATH.name}")
        else:
            raise FileNotFoundError(
                f"Extraction succeeded but no CSV found in {DATA_DIR}. "
                f"Archive layout may have changed. "
                f"Try downloading manually from {BACKUP_URL}"
            )

    print(f"Data saved to {CSV_PATH}")
    return CSV_PATH


if __name__ == "__main__":
    path = download_cfpb()
    print(f"CSV at: {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
