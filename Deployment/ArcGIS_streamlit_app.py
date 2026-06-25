import streamlit as st
import os
import json
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from arcgis.gis import GIS
from arcgis.features import FeatureLayer
import tempfile
import time
import requests

# ==========================================
# CONFIGURATION & API KEY INITIALIZATION
# ==========================================
MODEL_ID = "gemini-3.5-flash"
GEMINI_PROJECT = "dataplus-godurham"

try:
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
        tfile.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        tfile.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tfile.name
            
    client = genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location="global"
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")

# ArcGIS SDK layer (still used for attachments)
@st.cache_resource
def get_arcgis_layer():
    try:
        gis = GIS(
            "https://dukeuniv.maps.arcgis.com",
            client_id=st.secrets["ARCGIS_CLIENT_ID"],
            client_secret=st.secrets["ARCGIS_CLIENT_SECRET"]
        )
        return FeatureLayer(st.secrets["FEATURE_LAYER_URL"], gis=gis)
    except Exception as e:
        st.error(f"Failed to securely authenticate with Duke ArcGIS: {e}")
        return None

layer = get_arcgis_layer()

# ==========================================
# PROMPTS & SCHEMAS
# ==========================================
PROMPT_PASS_1 = """
You are an expert visual analyst evaluating a sequence of Street View images for a transit stop accessibility inventory.

CRITICAL INSTRUCTION: You must follow a strict TWO-PATH logical workflow. Attempt Path A first. Only proceed to Path B if Path A fails.

=== PATH A: SINGLE-IMAGE FAST-TRACK ===
Scan all provided images for clear, explicit transit infrastructure (a dedicated GoDurham bus stop sign, a bus shelter, or a dedicated transit bench).
(Note: Parked buses, generic yellow poles, utility poles, and bike racks DO NOT count).

If you find a clear view of the transit infrastructure AND the boarding area in ONE single image:
1. Set bus_stop_visible to "Yes".
2. Set best_view to the name of that specific winning image.
3. Classify ALL features (stop_surface, landing_type, amenities) using ONLY that single image. 
STILL: Synthesize all of the images to search for a cross walk. 
Stop here and output your JSON.

=== PATH B: PANORAMIC SYNTHESIS ===
If no single image provides a perfect view, you must synthesize the visual evidence from ALL images combined to evaluate the continuous environment.

Evaluate the synthesized environment and choose ONE of the following outcomes:

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama proves a bus stop exists (e.g., the transit sign is in the far-left image, but the concrete landing pad and bench are in the mid-left image).
- Action: Set bus_stop_visible to "Yes". Set best_view to the image containing the transit sign or the clearest part of the boarding pad. Synthesize the environment to count all features accurately. DO NOT flag for manual review.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360/panoramic area. There is absolutely no transit infrastructure (no bus sign, no shelter). You only see general street features (sidewalks, grass, utility poles).
- Action: Set bus_stop_visible to "No". Classify whatever generic features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction completely blocks the view of the curb. Do NOT assume a stop exists just because a bus is parked there.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."

Important: 
Confidence Scores: bus_stop_visibility, shelter_present, bench_present. trash_can_present
These attributes are extremely important. They will show up as decimal values between 0.0 and 1.0 that will accuratly
Represent your confidence in the presence of each of these attributes. It is important that give your honest rating and actually include
decimal values in between 0.0 and 1.0 in a reproducible way as we will experiment around with different thresholds for a final classifcation.
Remember this when assinging your confidence scores. 

=== DEFINITIONS ===
stop_name: EXACT value saved in variable: "name"
lattidude: EXACT saved in variable: "lattitude"
longitude: EXACT saved in variable: "longitude"
1. shelter_number: Total integer count across ALL images.
2. bench_number: Total integer count across ALL images. If there is a shelter present there is at least one.
3. trash_can_number: Total integer count of ANY visible public trash cans. 
4. stop_surface: "Grass" or "Concrete"
5. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
6. sidewalk_connection: "Yes", "No", or "NA"
7. landing_pad: "Two_doors", "One_door", or "NA"
8. cross_walk: "Yes", or "No"
9. street_lighting: "Yes" or "No"

Return only JSON matching the schema.
"""

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {"type": "string"},
        "stop_name": {"type": "string"},
        "selected_image_filename": {"type": "string"},
        "lattitude": {"type": "number"},
        "longitude": {"type": "number"},
        "best_view": {"type": "string"},
        "bus_stop_visible": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
        "bus_stop_visibility_confidence": {"type": "number"},
        "shelter_present": {"type": "number"},
        "shelter_number": {"type": "integer"},
        "bench_present": {"type": "number"},
        "bench_number": {"type": "integer"},
        "trash_can_present": {"type": "number"},
        "trash_can_number": {"type": "integer"},
        "stop_surface": {"type": "string", "enum": ["Grass", "Concrete"]},
        "landing_type": {"type": "string", "enum": ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"]},
        "sidewalk_connection": {"type": "string", "enum": ["Yes", "No", "NA"]},
        "landing_pad": {"type": "string", "enum": ["Two_doors", "One_door", "NA"]},
        "cross_walk": {"type": "string", "enum": ["Yes", "No"]},
        "street_lighting": {"type": "string", "enum": ["Yes", "No"]},
        "date": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": [
        "stop_id", "stop_name", "selected_image_filename", "lattitude", "longitude", "best_view",
        "bus_stop_visible", "bus_stop_visibility_confidence",
        "stop_surface", "landing_type", "sidewalk_connection", "landing_pad",
        "shelter_number", "shelter_present", "bench_number", "bench_present",
        "trash_can_number", "trash_can_present", "date", "street_lighting", "notes"
    ],
    "additionalProperties": False
}

# ==========================================
# ARCGIS TOKEN MANAGEMENT
# ==========================================
_token_cache = {"value": None, "expires_at": 0}

def get_arcgis_token() -> str:
    """Returns a valid token using secrets, auto-refreshing before expiry."""
    if _token_cache["value"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["value"]

    resp = requests.post(
        "https://www.arcgis.com/sharing/rest/oauth2/token",
        data={
            "client_id": st.secrets["ARCGIS_CLIENT_ID"],      # fixed: was using client_secret twice
            "client_secret": st.secrets["ARCGIS_CLIENT_SECRET"],
            "grant_type": "client_credentials",
            "f": "json",
        },
    ).json()

    if "access_token" not in resp:
        raise RuntimeError(f"Token fetch failed: {resp}")

    _token_cache["value"] = resp["access_token"]
    _token_cache["expires_at"] = time.time() + resp.get("expires_in", 7200)
    return _token_cache["value"]

# ==========================================
# BACKEND ARCGIS SYNC & ATTACHMENT PIPELINE
# ==========================================
def push_to_arcgis_server(stop_id: str, gemini_results: dict, uploaded_file) -> tuple:
    """
    Drop-in replacement with original 3-argument signature.
    Pulls layer_url and token internally — no call-site changes needed.
    """
    try:
        layer_url = st.secrets["FEATURE_LAYER_URL"]
        token = get_arcgis_token()

        # 1. Query for the OBJECTID
        query_resp = requests.get(
            f"{layer_url}/query",
            params={
                "where": f"stop_id='{stop_id}'",
                "outFields": "OBJECTID",
                "f": "json",
                "token": token,
            },
        ).json()

        if "error" in query_resp:
            return False, f"Query failed: {query_resp['error']}"
        
        st.write("Raw query response:", query_resp)

        features = query_resp.get("features", [])
        if not features:
            return False, f"Stop ID {stop_id} missing on target map service layer."

        object_id = features[0]["attributes"]["OBJECTID"]

        # 2. Build update payload — plain dicts, zero SDK objects
        update_payload = [{
            "attributes": {
                "OBJECTID":             int(object_id),
                "bus_stop_visible":     str(gemini_results.get("bus_stop_visible", "Yes")),
                "shelter_number":       int(gemini_results.get("shelter_number", 0)),
                "bench_number":         int(gemini_results.get("bench_number", 0)),
                "trash_can_number":     int(gemini_results.get("trash_can_number", 0)),
                "stop_surface":         str(gemini_results.get("stop_surface", "Concrete")),
                "landing_type":         str(gemini_results.get("landing_type", "Paved")),
                "sidewalk_connection":  str(gemini_results.get("sidewalk_connection", "Yes")),
                "landing_pad":          str(gemini_results.get("landing_pad", "Two_doors")),
                "notes":                str(gemini_results.get("notes", "")),
            }
        }]

        # 3. Apply the edit
        edit_resp = requests.post(
            f"{layer_url}/applyEdits",
            data={
                "updates": json.dumps(update_payload),
                "f": "json",
                "token": token,
            },
        ).json()

        results = edit_resp.get("updateResults", [])
        if not results or not results[0].get("success"):
            err = results[0].get("error", {}) if results else edit_resp
            return False, f"REST applyEdits failed: {err}"

        # 4. Attachment upload (REST multipart — no SDK layer object needed)
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, "rb") as f_attach:
                attach_resp = requests.post(
                    f"{layer_url}/{object_id}/addAttachment",
                    params={"token": token, "f": "json"},
                    files={"attachment": (uploaded_file.name, f_attach, uploaded_file.type)},
                ).json()

            os.unlink(tmp_file_path)

            if not attach_resp.get("addAttachmentResult", {}).get("success"):
                # Non-fatal: attribute sync succeeded, just warn about attachment
                return True, f"Stop {stop_id} synced. ⚠️ Attachment upload failed: {attach_resp}"

        return True, f"Perfect Sync! Stop {stop_id} elements updated."

    except Exception as e:
        return False, f"ArcGIS Live Data Stream Exception: {e}"
    

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(page_title="GoDurham Map Integration System", layout="centered")

st.title("🚌 GoDurham Live Map Server Sync Hub")
st.write("Upload a live field photo to evaluate it with Gemini and push the outputs straight to the city's ArcGIS infrastructure inventory database.")

stop_id = st.text_input("Enter Stop ID (e.g., 5238):")
uploaded_file = st.file_uploader("Upload Bus Stop Image", type=["jpg", "jpeg", "png"])

if st.button("Classify & Sync with ArcGIS Server"):
    if not stop_id:
        st.warning("Please specify a Stop ID to target your feature rows.")
    elif uploaded_file is None:
        st.warning("Please upload a field visualization photo file.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Processing Staging View for Stop ID: {stop_id}", use_container_width=True)
        
        with st.spinner('Running Multimodal Classification & Mapping Pipeline...'):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[PROMPT_PASS_1, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1
                    )
                )
                
                result_json = json.loads(response.text)
                result_json["stop_id"] = str(stop_id)
                result_json["selected_image_filename"] = str(uploaded_file.name)
                
                # Call site unchanged — still 3 arguments
                sync_success, sync_msg = push_to_arcgis_server(stop_id, result_json, uploaded_file)
                
                if sync_success:
                    st.success(sync_msg)
                else:
                    st.warning(f"AI Eval complete, but Map Sync missed: {sync_msg}")
                
                st.subheader("Generated Inventory Payload:")
                st.json(result_json)
                
            except Exception as e:
                st.error(f"Operational Pipeline Disruption: {e}")