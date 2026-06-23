import streamlit as st
import os
import json
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION & API KEY INITIALIZATION
# ==========================================
MODEL_ID = "gemini-3.5-flash"
GEMINI_PROJECT = "dataplus-godurham"

# Helper function matching your pipeline's local key reader logic
def load_api_key(path: str) -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"Could not find API key file: {key_path}")
    return key_path.read_text(encoding="utf-8").strip()

# Initialize the Gemini Client for your enterprise project
try:
    # 1. Point the client to Vertex AI without passing an API key string
    client = genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location="global"
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")

# Paste your full PROMPT_PASS_1 here
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

PROMPT_PASS_2 = """
You are an expert visual analyst evaluating a 360-degree panoramic sequence of a transit stop location. 
YOU ARE A PRECISE MACHINE THAT PRIORITIZES REPRODUCIBILITY. FOLLOW EVER LINE AS IT COMES UP, THE PROMPT IS MENT TO BE FOLLOWED IN ORDER.

INSTRUCTIONS:
You must SYNTHESIZE the visual evidence from ALL images combined to evaluate the continuous environment.

Classification Refresher:
Scan the provided images for clear, explicit transit infrastructure. If ANY of these components are found then the stop EXISTS.
1. A GoDurham bus stop sign (often attached to wooden utility poles or metal posts). Depending on the view you may be able to make out "Bus Stop" written on the stop if close enough and have a appropriate angle.
2. A bus shelter, will likely include a bench situated inside of it.
3. A public bench either included with the bus shelter, or in close proximity to a bus sign or bus shelter. 
4. A public trash can adjacent to the road that is near where pedestrians would walk.
(Note: Parked buses will obstruct view, bike racks, fire hydrants, and BARE utility poles or WITHOUT the up-right rectangle GoDurham bus sign DO NOT count).

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama has ANY of the previously listed classification refreshers(1-4) are present throughout the entire synthesis. THEY DO NOT NEED TO BE IN CLOSE PROXIMITY TO EACH OTHER.
- Action: Set bus_stop_visible to "Yes". Set best_view to the EXACT string identifier of the image containing any of the components or the best representation of a component. (e.g., "n", "sw", "right"). Synthesize the environment across all images to count features.
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification, be sure to include the bus stop components in this note.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360 area and found ZERO of the components. (There is absolutely no sign, no shelter, no bench, AND no trash can).
- Action: Set bus_stop_visible to "No". Classify features you can see (sidewalks, grass). You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification, be sure to include the bus stop components in this note.

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction blocks the view of the curb across ALL relevant images.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification

IMPORTANT: 
Confidence Scores: bus_stop_visibility, shelter_present, bench_present. trash_can_present
These attributes are extremely important. They will show up as decimal values between 0.0 and 1.0 that will accuratly
Represent your confidence in the presence of each of these attributes. It is important that give your HONEST rating and actually include
decimal values in between 0.0 and 1.0 in a reproducible way as we will experiment around with different thresholds for a final classifcation.
Remember this when assinging your confidence scores. 

DEFINITIONS:
stop_name: EXACT value saved in variable: "name"
lattidude: EXACT saved in variable: "lattitude"
longitude: EXACT saved in variable: "longitude"
1. shelter_number: Total integer count across ALL images.
2. bench_number: Total integer count across ALL images, if there is a shelter present there is at least one.
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
        "best_view": {
            "type": "string",
            "description": "The exact view name (e.g., left, right, center, front, front_right) that best shows the stop."
        },
        "bus_stop_visible": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
        "bus_stop_visibility_confidence": {
            "type": "number",
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a Bus stop. "
            "1.0 means with ALL certainty a Bus Stop EXISTS "
            "0.0 means there is no possibility of a Bus Stop AT ALL"
            
        },
        
        "shelter_present": {
            "type": "number", 
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a shelter. "
            "1.0 means with ALL certainty a shelter EXISTS "
            "0.0 means there is no possibility of a shelter AT ALL"
        },
        "shelter_number": {"type": "integer"},

        "bench_present": {
            "type": "number", 
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a bench. "
            "1.0 means with ALL certainty a bench EXISTS "
            "0.0 means there is no possibility of a bench AT ALL"
        },
        "bench_number": {"type": "integer"},

        "trash_can_present": {
            "type": "number", 
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a trash can. "
            "1.0 means with ALL certainty a trash can EXISTS "
            "0.0 means there is no possibility of a trash can AT ALL"
        },
        "trash_can_number": {"type": "integer"},
        
        "stop_surface": {"type": "string", "enum": ["Grass", "Concrete"]},
        "landing_type": {"type": "string", "enum": ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"]},
        "sidewalk_connection": {"type": "string", "enum": ["Yes", "No", "NA"]},
        "landing_pad": {"type": "string", "enum": ["Two_doors", "One_door", "NA"]},
        "cross_walk": {"type": "string", "enum": ["Yes", "No"]},
        "street_lighting": {"type": "string", "enum": ["Yes", "No"]},
        "date": {
            "type": "string",
            "description": "The exact year and month the image was taken (e.g., 2026-1, 2025-3) explicitly stated in the file name"
        },
        "notes": {"type": "string"},
    },
    "required": [
        "stop_id", "stop_name", "selected_image_filename", "lattitude", "longitude", "best_view", "bus_stop_visible", "bus_stop_visibility_confidence", 
        "stop_surface", "landing_type", "sidewalk_connection", "landing_pad", 
        "shelter_number", "shelter_present",
        "bench_number", "bench_present",  
        "trash_can_number", "trash_can_present", "date",
        "street_lighting", "notes"
    ],
    "additionalProperties": False
}

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(page_title="GoDurham Bus Stop Classifier", layout="centered")

st.title("🚌 GoDurham Bus Stop Classifier")
st.write("Upload a field photo of a bus stop to automatically classify its features and generate the JSON inventory data.")

# Input fields for the user
stop_id = st.text_input("Enter Stop ID (e.g., 5064):")
uploaded_file = st.file_uploader("Upload Bus Stop Image", type=["jpg", "jpeg", "png"])

# Action Button
if st.button("Classify Stop"):
    if not stop_id:
        st.warning("Please enter a Stop ID.")
    elif uploaded_file is None:
        st.warning("Please upload an image.")
    else:
        # 1. Display the uploaded image on the screen
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Stop ID: {stop_id}", use_container_width=True)
        
        # 2. Run the API call with a visual loading spinner
        with st.spinner('Analyzing image with Gemini...'):
            try:
                # Call the Gemini API directly with the PIL Image
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[{PROMPT_PASS_1}, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1 # Keep it low for strict, analytical outputs
                    )
                )
                
                # 3. Parse and display the result
                result_json = json.loads(response.text)
                
                st.success("Classification Complete!")
                
                # Display the data nicely in an interactive JSON block
                st.subheader("Extracted Inventory Data:")
                st.json(result_json)
                
                # Add a button to let the user download the raw JSON file
                st.download_button(
                    label="Download JSON Data", 
                    data=response.text, 
                    file_name=f"{stop_id}_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")