import os
import time
import math
import requests
import pandas as pd

stops = pd.read_csv("Altered 2026 GoDurham Bus Stop List.csv")

stops = stops.sort_values("Stop Code")

stops = stops.head(20)


print(f"Running on {len(stops)} stops")

with open(".api/api_key.txt", "r") as f:
    api_key = f.read().strip()

os.makedirs("images_metadata", exist_ok=True)

def clean_filename(text):
    text = str(text).strip()
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']
    for char in bad_chars:
        text = text.replace(char, "")
    text = text.replace(" ", "_")
    return text

def get_metadata(lat, lon):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "radius": 25,
        "key": api_key
    }

    response = requests.get(url, params=params)
    return response.json()


def calculate_heading(from_lat, from_lon, to_lat, to_lon):
    """
    Calculates compass direction from the Street View camera
    to the actual bus stop.
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


for _, row in stops.iterrows():
    stop_code = row["Stop Code"]
    stop_name = row["Stop Name"]
    bus_lat = row["Latitude"]
    bus_lon = row["Longitude"]

    metadata = get_metadata(bus_lat, bus_lon)

    if metadata.get("status") != "OK":
        print(f"Skipping {stop_code}: no Street View found")
        continue

    pano_lat = metadata["location"]["lat"]
    pano_lon = metadata["location"]["lng"]
    image_date = metadata.get("date", "unknown-date")
    print(f"{stop_code}: Street View date = {image_date}")

    heading = calculate_heading(
        from_lat=pano_lat,
        from_lon=pano_lon,
        to_lat=bus_lat,
        to_lon=bus_lon
    )

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "pano": metadata["pano_id"],
        "heading": heading,
        "pitch": 0,
        "fov": 60,
        "key": api_key
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Failed {stop_code}: image request error")
        continue

    safe_stop_name = clean_filename(stop_name)
    filename = f"images_metadata/{stop_code}_{safe_stop_name}_{image_date}_heading-{round(heading)}.jpg"

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved {filename} - {stop_name} - heading {round(heading, 1)}")

    time.sleep(0.1)

print("Done.")