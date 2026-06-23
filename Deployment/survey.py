import streamlit as st
import datetime

st.set_page_config(page_title="Bus Stop Audit", page_icon="🚏")

st.title("🚏 Bus Stop Audit Form")
st.write("Log the condition and location of the bus stop.")

# Create the form
with st.form("bus_stop_form", clear_on_submit=False):
    st.subheader("1. Stop Identification")
    bus_stop_code = st.text_input("Bus Stop Code", placeholder="e.g., BS-4029")
    
    # Use columns to put Lat/Lon side-by-side
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", format="%.6f", value=0.0)
    with col2:
        lon = st.number_input("Longitude", format="%.6f", value=0.0)
        
    st.subheader("2. Audit Details")
    # Defaults to today's date automatically
    audit_date = st.date_input("Date of Audit", datetime.date.today())
    
    # Allow multiple image uploads (great for mobile users, it will open their camera roll)
    photos = st.file_uploader(
        "Upload Bus Stop Photos", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )
    
    submit_button = st.form_submit_button("Submit Audit")

# Handle the data when the user clicks submit
if submit_button:
    # Basic validation to ensure they didn't leave critical fields blank
    if not bus_stop_code:
        st.error("Please enter a Bus Stop Code.")
    elif not photos:
        st.error("Please upload at least one photo.")
    elif lat == 0.0 and lon == 0.0:
        st.warning("Warning: Coordinates are set to 0.0, 0.0.")
    else:
        # Success block
        st.success("Audit submitted successfully!")
        
        # Display a summary of the captured data
        st.write("### Captured Data Summary:")
        st.write(f"**Stop Code:** {bus_stop_code}")
        st.write(f"**Date:** {audit_date}")
        st.write(f"**Coordinates:** {lat}, {lon}")
        st.write(f"**Total Photos:** {len(photos)}")
        
        # Display the first photo as a quick preview
        st.image(photos[0], caption="Preview of first photo", use_container_width=True)
        
        # NOTE: At this point, you would write the text data to a database
        # and save the `photos` objects to a cloud bucket like AWS S3 or Google Cloud Storage.