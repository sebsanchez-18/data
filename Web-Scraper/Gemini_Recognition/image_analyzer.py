import os
import re
import csv
import json
import shutil
import mimetypes
from pathlib import Path
from collections import defaultdict

from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("images_metadata")          # folder containing all left/center/right images
FINAL_IMAGES_DIR = Path("final_images")
OUTPUT_JSON = Path("bus_stop_results.json")
OUTPUT_CSV = Path("bus_stop_results.csv")

MODEL = "gemini-3.5-flash"

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


# ============================================================
# YOUR PARSER
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

Each bus stop has three image views:
- left
- center
- right

First, determine whether the image trio appears to show a bus stop or bus stop area.

Set bus_stop_visible:
- "Yes" if at least one of the three images clearly shows a bus stop sign, boarding/landing area, shelter, bench, route sign, bus stop pole, or obvious bus stop zone.
- "No" if none of the three images appear to show a bus stop or relevant stop area.
- "Unclear" if the scene may show the stop area but visual evidence is limited, blocked, blurry, too far away, or ambiguous.

Set bus_stop_visibility_confidence from 0.0 to 1.0 based on how confident you are in the bus_stop_visible label.

Then choose which image gives the best overall view of the bus stop.
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
Allowed values: "left", "center", "right"

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
            "type": "string",
            "enum": ["left", "center", "right"],
        },
        "selected_image_filename": {
            "type": "string"
        },

        # New bus stop visibility check
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
# GEMINI CALL
# ============================================================

def analyze_stop(stop_id: str, views: dict) -> dict:
    left_path = views["left"]
    center_path = views["center"]
    right_path = views["right"]

    contents = [
        PROMPT,
        f"""
Stop ID: {stop_id}

The next three images are the left, center, and right views of the same bus stop.

Choose the best_view from these three images.
Then classify the stop using that selected image.
Set selected_image_filename to the filename of the chosen image.
""",
        "LEFT VIEW:",
        make_image_part(left_path),
        f"Left filename: {left_path.name}",
        "CENTER VIEW:",
        make_image_part(center_path),
        f"Center filename: {center_path.name}",
        "RIGHT VIEW:",
        make_image_part(right_path),
        f"Right filename: {right_path.name}",
    ]

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

    selected_view = result["best_view"]
    selected_path = views[selected_view]

    result["stop_id"] = stop_id
    result["selected_image_filename"] = selected_path.name

    destination = FINAL_IMAGES_DIR / selected_path.name
    shutil.copy2(selected_path, destination)

    result["final_image_path"] = str(destination)

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
            result = analyze_stop(stop_id, views)
            results.append(result)

            # Save after every stop so progress is not lost.
            save_json(results)
            write_csv(results)

            print(
                f"Done stop {stop_id}: selected {result['best_view']} "
                f"({result['selected_image_filename']})"
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