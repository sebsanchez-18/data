import os
import time
import math
import argparse
import requests
import pandas as pd


# ---------------------------------------------------------
# 1. GLOBAL SETUP
# ---------------------------------------------------------

CSV_PATH = "Altered 2026 GoDurham Bus Stop List.csv"
API_KEY_PATH = ".api/api_key.txt"
DEFAULT_OUTPUT_DIR = "images_metadata_6headings"


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def clean_filename(text):
    text = str(text).strip()
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']

    for char in bad_chars:
        text = text.replace(char, "")

    text = text.replace(" ", "_")
    return text


def normalize_stop_code(stop_code):
    """
    Ensures stop codes are always four digits.

    Example:
        17   -> 0017
        123  -> 0123
        1234 -> 1234
    """

    return str(stop_code).strip().zfill(4)


def load_stops(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find stops CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["Stop Code", "Stop Name", "Latitude", "Longitude"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def load_api_key(api_key_path):
    if not os.path.exists(api_key_path):
        raise FileNotFoundError(f"Could not find API key file: {api_key_path}")

    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    if not api_key:
        raise ValueError(f"API key file is empty: {api_key_path}")

    return api_key


def get_metadata(lat, lon, api_key):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    params = {
        "location": f"{lat},{lon}",
        "radius": 25,
        "key": api_key,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def calculate_heading(from_lat, from_lon, to_lat, to_lon):
    """
    Calculates compass direction from the Street View camera to the bus stop.
    """

    from_lat = math.radians(from_lat)
    from_lon = math.radians(from_lon)
    to_lat = math.radians(to_lat)
    to_lon = math.radians(to_lon)

    d_lon = to_lon - from_lon

    x = math.sin(d_lon) * math.cos(to_lat)
    y = (
        math.cos(from_lat) * math.sin(to_lat)
        - math.sin(from_lat) * math.cos(to_lat) * math.cos(d_lon)
    )

    heading = math.degrees(math.atan2(x, y))
    return (heading + 360) % 360


# ---------------------------------------------------------
# 3. MAIN SCRAPER FUNCTION
# ---------------------------------------------------------

def fetch_stop_images(target_stop_code, stops_df, api_key, output_dir):
    """
    Fetches a 6-image Street View panorama sweep for a specific bus stop code.
    """

    target_stop_code = normalize_stop_code(target_stop_code)

    os.makedirs(output_dir, exist_ok=True)

    stop_codes_normalized = (
        stops_df["Stop Code"]
        .astype(str)
        .str.strip()
        .str.zfill(4)
    )

    stop_data = stops_df[stop_codes_normalized == target_stop_code]

    if stop_data.empty:
        print(f"Error: Stop code {target_stop_code} not found in the CSV.")
        return False

    row = stop_data.iloc[0]

    stop_name = row["Stop Name"]
    bus_lat = row["Latitude"]
    bus_lon = row["Longitude"]

    print(f"Processing Stop {target_stop_code}: {stop_name}")

    metadata = get_metadata(bus_lat, bus_lon, api_key)

    if metadata.get("status") != "OK":
        print(f"  -> Skipping. Google says: {metadata}")
        return False

    pano_lat = metadata["location"]["lat"]
    pano_lon = metadata["location"]["lng"]
    pano_id = metadata["pano_id"]
    image_date = metadata.get("date", "unknown-date")

    print(f"  -> Street View date = {image_date}")
    print(f"  -> Pano ID = {pano_id}")

    heading = calculate_heading(
        from_lat=pano_lat,
        from_lon=pano_lon,
        to_lat=bus_lat,
        to_lon=bus_lon,
    )

    print(f"  -> Heading toward stop = {round(heading, 1)}")

    safe_stop_name = clean_filename(stop_name)

    sweep_offsets = {
        "far_left": -75,
        "mid_left": -45,
        "slight_left": -15,
        "slight_right": 15,
        "mid_right": 45,
        "far_right": 75,
    }

    saved_count = 0

    for view_name, offset in sweep_offsets.items():
        sweep_heading = (heading + offset) % 360
        rounded_heading = round(sweep_heading)

        url = "https://maps.googleapis.com/maps/api/streetview"

        params = {
            "size": "640x640",
            "pano": pano_id,
            "heading": sweep_heading,
            "pitch": 0,
            "fov": 60,
            "key": api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            if "image" not in content_type.lower():
                print(f"  -> Failed {view_name}: response was not an image")
                continue

        except requests.RequestException as e:
            print(f"  -> Failed {view_name}: {e}")
            continue

        filename = os.path.join(
            output_dir,
            f"{target_stop_code}_{safe_stop_name}_{image_date}_"
            f"{view_name}_heading-{rounded_heading}.jpg",
        )

        with open(filename, "wb") as f:
            f.write(response.content)

        saved_count += 1

        print(
            f"  -> Saved {view_name} "
            f"(Heading {round(sweep_heading, 1)})"
        )

        time.sleep(0.1)

    print(
        f"Finished processing stop {target_stop_code}. "
        f"Saved {saved_count} images.\n"
    )

    return saved_count > 0


# ---------------------------------------------------------
# 4. COMMAND LINE ENTRYPOINT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch 6 Street View heading images for one GoDurham bus stop."
    )

    parser.add_argument(
        "--stop-id",
        required=True,
        type=str,
        help="Four-digit bus stop code to scrape.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where scraped images should be saved.",
    )

    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help="Path to the bus stop CSV file.",
    )

    parser.add_argument(
        "--api-key",
        default=API_KEY_PATH,
        help="Path to the Google Street View API key file.",
    )

    args = parser.parse_args()

    stops_df = load_stops(args.csv)
    api_key = load_api_key(args.api_key)

    success = fetch_stop_images(
        target_stop_code=args.stop_id,
        stops_df=stops_df,
        api_key=api_key,
        output_dir=args.output_dir,
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()