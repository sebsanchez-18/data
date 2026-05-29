import os
import re
import csv
import sys
import json
import shutil
import mimetypes
import subprocess
from pathlib import Path
from collections import defaultdict

from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("images_metadata")          # folder containing original left/center/right images
FINAL_IMAGES_DIR = Path("final_images")
OUTPUT_JSON = Path("bus_stop_results.json")
OUTPUT_CSV = Path("bus_stop_results.csv")

MODEL = "gemini-3.5-flash"

# Script used to get more images when a stop is labeled "No"
WEBSCRAPE_SCRIPT = Path("updated_six_headings.py")

# Folder where webscrape_6headings.py saves new images.
# Change this if your scraper saves somewhere else.
SCRAPED_IMAGES_DIR = Path("images_metadata_6headings")

# Only scrape more images for these first-pass labels.
# If you also want to rescrape ambiguous stops, use {"No", "Unclear"}.
RESCRAPE_VISIBILITY_LABELS = {"No"}

# Safety cap so a bad scrape does not upload tons of images to Gemini.
MAX_EXTRA_IMAGES_PER_STOP = 12


# If using Google Cloud / Vertex / Agent Platform:
def load_api_key(path: str) -> str:
    key_path = Path(path)

    if not key_path.exists():
        raise FileNotFoundError(
            f"Could not find API key file: {key_path}"
        )

    api_key = key_path.read_text(encoding="utf-8").strip()

    if not api_key:
        raise ValueError(f"API key file is empty: {key_path}")

    return api_key


client = genai.Client(
    vertexai=True,
    project="dataplus-godurham",
    location="global",
)


FINAL_IMAGES_DIR.mkdir(exist_ok=True)
SCRAPED_IMAGES_DIR.mkdir(exist_ok=True)


# ============================================================
# PARSER
# ============================================================

def parse_stop_id_and_view(path: Path):
    filename = path.name.lower()

    match = re.match(r"^(\d{4})", filename)
    if not match:
        raise ValueError(
            f"Filename must start with a four-digit stop code: {path.name}"
        )

    stop_id = match.group(1)

    # Detect view only from the actual view tag before "_heading"
    view_match = re.search(r"_(left|center|centre|right)_heading", filename)

    if not view_match:
        raise ValueError(
            f"Could not determine view from filename: {path.name}. "
            "Expected pattern like '_left_heading', '_center_heading', or '_right_heading'."
        )

    view = view_match.group(1)

    if view == "centre":
        view = "center"

    return stop_id, view


# ============================================================
# GROUP IMAGES BY 4-DIGIT STOP CODE
# ============================================================

def group_images_by_stop(input_dir: Path):
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(input_dir.rglob(ext))

    grouped = defaultdict(dict)

    for path in image_paths:
        stop_id, view = parse_stop_id_and_view(path)

        if view in grouped[stop_id]:
            raise ValueError(
                f"Duplicate {view} image for stop {stop_id}: "
                f"{grouped[stop_id][view].name} and {path.name}"
            )

        grouped[stop_id][view] = path

    complete_groups = {}
    incomplete_groups = {}

    for stop_id, views in grouped.items():
        if all(v in views for v in ["left", "center", "right"]):
            complete_groups[stop_id] = views
        else:
            incomplete_groups[stop_id] = views

    return complete_groups, incomplete_groups


# ============================================================
# GEMINI PROMPT
# ============================================================

PROMPT = """
You are analyzing bus stop images for transit stop accessibility inventory.

Each bus stop may have multiple image views.
The images will be provided with labels such as:
- left
- center
- right
- heading_0
- heading_60
- heading_120
- scraped_01
- scraped_02

First, compare all provided images.

Determine whether the image set appears to show a bus stop or bus stop area.

Set bus_stop_visible:
- "Yes" if at least one image clearly shows a bus stop sign, boarding/landing area, shelter, bench, route sign, bus stop pole, or obvious bus stop zone.
- "No" if none of the images appear to show a bus stop or relevant stop area.
- "Unclear" if the scene may show the stop area but visual evidence is limited, blocked, blurry, too far away, or ambiguous.

Set bus_stop_visibility_confidence from 0.0 to 1.0 based on how confident you are in the bus_stop_visible label.

Then choose which single image gives the best overall view of the bus stop.
The best view should show the boarding/landing area, road edge or curb,
sidewalk if present, and nearby amenities such as shelter, bench, trash can,
and lighting.

Then classify the bus stop using the best selected image only.

Use only visible evidence. Do not guess hidden features.
If a field is not applicable, use "NA".
If uncertain, choose the most visually supported option and lower the confidence.
Count only clearly visible objects at or immediately around the bus stop.

If bus_stop_visible is "No" or "Unclear":
- still choose the best available view
- still classify visible attributes as best as possible
- explain the concern in notes

Definitions:

1. stop_surface
Allowed values: "Grass", "Concrete"

Choose "Grass" when the surface immediately next to the road/curb where a rider
would stand or board is mostly grass, dirt, or unpaved ground.

Choose "Concrete" when that surface is mostly concrete, pavement, asphalt,
or another hard paved surface.

2. landing_type
Allowed values: "Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"

Choose "Paved" when the bus stop landing/standing area next to the road is paved,
usually concrete, asphalt, or a paved sidewalk/road shoulder.

Choose "Unpaved" when the landing/standing area next to the road is grass, dirt,
gravel, or otherwise unpaved and there is no nearby sidewalk forming part of the stop area.

Choose "Unpaved_Grass_Strip_And_Sidewalk" when the area next to the road is
grass/unpaved but there is a sidewalk nearby or behind it, creating a grass strip
between the road and sidewalk.

3. sidewalk_connection
Allowed values: "Yes", "No", "NA"

Choose "Yes" if there is a paved path, curb cut, concrete pad, sidewalk, or
continuous paved surface connecting the pedestrian area to the road/curb.

Also choose "Yes" when the stop area is concrete/paved in its entirety from the
sidewalk or standing area to the curb.

Choose "No" if a sidewalk is visible but the rider would have to cross grass,
dirt, gravel, or another unpaved surface to reach the road/curb.

Choose "NA" if there is no sidewalk or pedestrian path visible.

4. landing_pad
Allowed values: "Two_doors", "One_door", "NA"

Only classify this when there is a usable paved boarding area with
sidewalk_connection = "Yes".

Choose "Two_doors" if the paved landing area appears long enough and positioned
to serve both the front and rear bus doors.

Choose "One_door" if the paved landing area appears to serve only one bus door.

Choose "NA" if there is no usable paved landing area, sidewalk_connection is
"No" or "NA", or the landing pad is not visible enough to decide.

5. shelter_number
Integer count.

0 means no visible shelter.
Count the number of bus shelters visible at or immediately around the stop.
A bench with a back panel but no roof should be counted as a bench, not a shelter.

6. bench_number
Integer count.

0 means no visible bench.
Count the number of benches visible at or immediately around the stop.

7. trash_can_number
Integer count.

0 means no visible trash can.
Count the number of trash cans visible at or immediately around the stop.

8. street_lighting
Allowed values: "Yes", "No"

Choose "Yes" if a streetlight, lamp post, or dedicated lighting fixture is visible near the stop.
Choose "No" if no lighting is visible near the stop.

9. best_view

Use the exact image label provided for the selected image.

Choose the image that gives the clearest and most complete view of the bus stop
and boarding area. Do not choose based only on image sharpness. Choose based on
usefulness for classification.

Return only JSON matching the schema.
"""


# ============================================================
# JSON SCHEMA
# ============================================================

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {
            "type": "string"
        },
        "best_view": {
            "type": "string"
        },
        "selected_image_filename": {
            "type": "string"
        },

        "bus_stop_visible": {
            "type": "string",
            "enum": ["Yes", "No", "Unclear"],
        },
        "bus_stop_visibility_confidence": {
            "type": "number",
        },

        "stop_surface": {
            "type": "string",
            "enum": ["Grass", "Concrete"],
        },
        "landing_type": {
            "type": "string",
            "enum": [
                "Paved",
                "Unpaved",
                "Unpaved_Grass_Strip_And_Sidewalk",
            ],
        },
        "sidewalk_connection": {
            "type": "string",
            "enum": ["Yes", "No", "NA"],
        },
        "landing_pad": {
            "type": "string",
            "enum": ["Two_doors", "One_door", "NA"],
        },
        "shelter_number": {
            "type": "integer",
            "minimum": 0,
        },
        "bench_number": {
            "type": "integer",
            "minimum": 0,
        },
        "trash_can_number": {
            "type": "integer",
            "minimum": 0,
        },
        "street_lighting": {
            "type": "string",
            "enum": ["Yes", "No"],
        },
        "confidence": {
            "type": "object",
            "properties": {
                "best_view": {
                    "type": "number"
                },
                "bus_stop_visible": {
                    "type": "number"
                },
                "stop_surface": {
                    "type": "number"
                },
                "landing_type": {
                    "type": "number"
                },
                "sidewalk_connection": {
                    "type": "number"
                },
                "landing_pad": {
                    "type": "number"
                },
                "shelter_number": {
                    "type": "number"
                },
                "bench_number": {
                    "type": "number"
                },
                "trash_can_number": {
                    "type": "number"
                },
                "street_lighting": {
                    "type": "number"
                },
            },
            "required": [
                "best_view",
                "bus_stop_visible",
                "stop_surface",
                "landing_type",
                "sidewalk_connection",
                "landing_pad",
                "shelter_number",
                "bench_number",
                "trash_can_number",
                "street_lighting",
            ],
        },
        "notes": {
            "type": "string"
        },
    },
    "required": [
        "stop_id",
        "best_view",
        "selected_image_filename",
        "bus_stop_visible",
        "bus_stop_visibility_confidence",
        "stop_surface",
        "landing_type",
        "sidewalk_connection",
        "landing_pad",
        "shelter_number",
        "bench_number",
        "trash_can_number",
        "street_lighting",
        "confidence",
        "notes",
    ],
}


# ============================================================
# IMAGE HELPERS
# ============================================================

def get_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {path}")
    return mime_type


def make_image_part(path: Path):
    return types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=get_mime_type(path),
    )


def image_sort_key(path: Path):
    return path.name.lower()


def find_scraped_images_for_stop(stop_id: str) -> list[Path]:
    """
    Finds newly scraped images for a stop.

    Expected possibilities:
    - scraped_images/1234/*.jpg
    - scraped_images/1234_heading_000.jpg
    - scraped_images/1234_*.png
    """

    if not SCRAPED_IMAGES_DIR.exists():
        return []

    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        # Case 1: files inside scraped_images/<stop_id>/
        stop_subdir = SCRAPED_IMAGES_DIR / stop_id
        if stop_subdir.exists():
            image_paths.extend(stop_subdir.rglob(ext))

        # Case 2: files directly under scraped_images/ beginning with stop_id
        image_paths.extend(SCRAPED_IMAGES_DIR.glob(f"{stop_id}*{ext[1:]}"))

    unique_paths = sorted(set(image_paths), key=image_sort_key)

    return unique_paths[:MAX_EXTRA_IMAGES_PER_STOP]


def call_webscrape_for_stop(stop_id: str) -> list[Path]:
    """
    Calls updated_six_headings.py for a stop ID, then returns scraped image paths.

    This matches:
        python updated_six_headings.py --stop-id 1234 --output-dir images_metadata_6headings
    """

    if not WEBSCRAPE_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find scraper script: {WEBSCRAPE_SCRIPT}")

    print(f"Calling scraper for stop {stop_id}...")

    subprocess.run(
        [
            sys.executable,
            str(WEBSCRAPE_SCRIPT),
            "--stop-id",
            str(stop_id),
            "--output-dir",
            str(SCRAPED_IMAGES_DIR),
        ],
        check=True,
    )

    scraped_images = find_scraped_images_for_stop(stop_id)

    if not scraped_images:
        print(f"No scraped images found for stop {stop_id} after scraper ran.")

    return scraped_images


def make_labeled_images_from_original_views(views: dict) -> list[tuple[str, Path]]:
    return [
        ("left", views["left"]),
        ("center", views["center"]),
        ("right", views["right"]),
    ]


def label_extra_images(extra_paths: list[Path]) -> list[tuple[str, Path]]:
    """
    Labels scraped images.

    If filename contains heading info, preserve it in the label.
    Otherwise use scraped_01, scraped_02, etc.
    """

    labeled = []

    used_labels = set()

    for i, path in enumerate(extra_paths, start=1):
        filename = path.stem.lower()

        heading_match = re.search(r"heading[_-]?(\d+)", filename)
        if heading_match:
            base_label = f"heading_{heading_match.group(1)}"
        else:
            base_label = f"scraped_{i:02d}"

        label = base_label
        suffix = 2

        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1

        used_labels.add(label)
        labeled.append((label, path))

    return labeled


# ============================================================
# LOGIC CLEANUP
# ============================================================

def enforce_logical_consistency(result: dict) -> dict:
    """
    Fix obvious contradictions after Gemini returns JSON.
    """

    if result["sidewalk_connection"] in ["No", "NA"]:
        result["landing_pad"] = "NA"

    if result["landing_type"] == "Unpaved" and result["sidewalk_connection"] != "Yes":
        result["landing_pad"] = "NA"

    if (
        result["stop_surface"] == "Concrete"
        and result["landing_type"] == "Paved"
        and result["sidewalk_connection"] == "NA"
    ):
        result["sidewalk_connection"] = "Yes"

    return result


# ============================================================
# GEMINI CALLS
# ============================================================

def analyze_labeled_images(stop_id: str, labeled_images: list[tuple[str, Path]]) -> dict:
    """
    Analyze any number of labeled images for one stop.

    Example:
        [
            ("left", Path(...)),
            ("center", Path(...)),
            ("right", Path(...)),
            ("heading_0", Path(...)),
            ("heading_60", Path(...)),
        ]
    """

    if not labeled_images:
        raise ValueError(f"No images provided for stop {stop_id}")

    contents = [
        PROMPT,
        f"""
Stop ID: {stop_id}

The following images are different views of the same possible bus stop area.

Each image has a label. Choose best_view using the exact label provided.
Set selected_image_filename to the filename of the chosen image.

Analyze only the selected best image when classifying stop_surface,
landing_type, sidewalk_connection, landing_pad, shelter_number,
bench_number, trash_can_number, and street_lighting.
""",
    ]

    for label, path in labeled_images:
        contents.extend([
            f"IMAGE LABEL: {label}",
            make_image_part(path),
            f"Filename for {label}: {path.name}",
        ])

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    result = json.loads(response.text)
    result = enforce_logical_consistency(result)

    result["stop_id"] = stop_id

    selected_label = result["best_view"]
    label_to_path = {label: path for label, path in labeled_images}

    if selected_label not in label_to_path:
        raise ValueError(
            f"Gemini selected best_view={selected_label}, but valid labels are "
            f"{list(label_to_path.keys())}"
        )

    selected_path = label_to_path[selected_label]

    result["selected_image_filename"] = selected_path.name

    destination = FINAL_IMAGES_DIR / selected_path.name

    # Avoid overwriting if original and scraped image share a filename.
    if destination.exists():
        destination = FINAL_IMAGES_DIR / f"{stop_id}_{selected_label}_{selected_path.name}"

    shutil.copy2(selected_path, destination)

    result["final_image_path"] = str(destination)

    return result


def analyze_stop(stop_id: str, views: dict) -> dict:
    """
    First-pass analysis using only original left/center/right images.
    """

    labeled_images = make_labeled_images_from_original_views(views)
    return analyze_labeled_images(stop_id, labeled_images)


def analyze_stop_with_scraped_images(
    stop_id: str,
    views: dict,
    scraped_images: list[Path],
) -> dict:
    """
    Second-pass analysis using original images plus scraped images.
    """

    original_labeled = make_labeled_images_from_original_views(views)
    extra_labeled = label_extra_images(scraped_images)

    all_labeled_images = original_labeled + extra_labeled

    result = analyze_labeled_images(stop_id, all_labeled_images)
    result["used_scraped_images"] = True
    result["scraped_image_count"] = len(scraped_images)

    return result


# ============================================================
# SAVE OUTPUTS
# ============================================================

def write_csv(results):
    fields = [
        "stop_id",
        "best_view",
        "selected_image_filename",
        "final_image_path",
        "bus_stop_visible",
        "bus_stop_visibility_confidence",
        "stop_surface",
        "landing_type",
        "sidewalk_connection",
        "landing_pad",
        "shelter_number",
        "bench_number",
        "trash_can_number",
        "street_lighting",
        "used_scraped_images",
        "scraped_image_count",
        "initial_bus_stop_visible",
        "initial_bus_stop_visibility_confidence",
        "initial_selected_image_filename",
        "notes",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            row = {field: r.get(field, "") for field in fields}
            writer.writerow(row)


def save_json(results):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    complete_groups, incomplete_groups = group_images_by_stop(INPUT_DIR)

    print(f"Complete stop trios found: {len(complete_groups)}")
    print(f"Incomplete stop groups found: {len(incomplete_groups)}")

    if incomplete_groups:
        print("\nIncomplete groups skipped:")
        for stop_id, views in incomplete_groups.items():
            print(f"{stop_id}: found views {list(views.keys())}")

    results = []
    already_done = set()

    # Resume if previous results exist.
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
            already_done = {r["stop_id"] for r in results}

    for stop_id, views in sorted(complete_groups.items()):
        if stop_id in already_done:
            print(f"Skipping already processed stop {stop_id}")
            continue

        try:
            print(f"Processing stop {stop_id}...")

            # First pass: original left/center/right only.
            initial_result = analyze_stop(stop_id, views)

            final_result = initial_result
            final_result["used_scraped_images"] = False
            final_result["scraped_image_count"] = 0
            final_result["initial_bus_stop_visible"] = initial_result["bus_stop_visible"]
            final_result["initial_bus_stop_visibility_confidence"] = initial_result[
                "bus_stop_visibility_confidence"
            ]
            final_result["initial_selected_image_filename"] = initial_result[
                "selected_image_filename"
            ]

            # Second pass: only when first pass says this is not a bus stop.
            if initial_result["bus_stop_visible"] in RESCRAPE_VISIBILITY_LABELS:
                print(
                    f"Stop {stop_id} was labeled bus_stop_visible="
                    f"{initial_result['bus_stop_visible']}. Scraping extra images..."
                )

                scraped_images = call_webscrape_for_stop(stop_id)

                if scraped_images:
                    print(
                        f"Re-analyzing stop {stop_id} with "
                        f"{len(scraped_images)} scraped images plus original views..."
                    )

                    final_result = analyze_stop_with_scraped_images(
                        stop_id=stop_id,
                        views=views,
                        scraped_images=scraped_images,
                    )

                    final_result["initial_bus_stop_visible"] = initial_result[
                        "bus_stop_visible"
                    ]
                    final_result["initial_bus_stop_visibility_confidence"] = initial_result[
                        "bus_stop_visibility_confidence"
                    ]
                    final_result["initial_selected_image_filename"] = initial_result[
                        "selected_image_filename"
                    ]

                    final_result["notes"] = (
                        f"Initial three-image analysis labeled this as "
                        f"{initial_result['bus_stop_visible']}. "
                        f"Scraped {len(scraped_images)} additional images and re-analyzed. "
                        f"Final notes: {final_result.get('notes', '')}"
                    )
                else:
                    final_result["notes"] = (
                        f"Initial analysis labeled this as {initial_result['bus_stop_visible']}. "
                        f"Scraper ran but no extra images were found. "
                        f"Original notes: {initial_result.get('notes', '')}"
                    )

            results.append(final_result)

            # Save after every stop so progress is not lost.
            save_json(results)
            write_csv(results)

            print(
                f"Done stop {stop_id}: final selected {final_result['best_view']} "
                f"({final_result['selected_image_filename']}), "
                f"bus_stop_visible={final_result['bus_stop_visible']}"
            )

        except Exception as e:
            print(f"FAILED stop {stop_id}: {e}")

    save_json(results)
    write_csv(results)

    print("\nFinished.")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Selected images copied to: {FINAL_IMAGES_DIR}")


if __name__ == "__main__":
    main()