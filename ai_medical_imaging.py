import os
from PIL import Image as PILImage
from dotenv import load_dotenv

# Load .env if present (server-side only)
load_dotenv()

# Note: This app never asks end users for the API key.
# The key is read only on the server from Streamlit secrets or GOOGLE_API_KEY env var.


from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.agent import RunOutput
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# ---- Server-side API key loading (no frontend key entry) ----
# Priority:
# 1) Streamlit secrets: st.secrets["GOOGLE_API_KEY"]
# 2) Environment variable: GOOGLE_API_KEY (can be provided via .env using python-dotenv)

GOOGLE_API_KEY = None
try:
    # st.secrets is available when using Streamlit secrets.toml
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    # Fall back to env var (e.g., loaded from .env at app startup)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

with st.sidebar:
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

medical_agent = None
if GOOGLE_API_KEY:
    medical_agent = Agent(
        model=Gemini(
            id="gemini-3.1-flash-lite",
            api_key=GOOGLE_API_KEY,
        ),
        tools=[DuckDuckGoTools()],
        markdown=True,
    )
else:
    st.error(
        "Server misconfigured: GOOGLE_API_KEY is not set. "
        "Set it via .env (GOOGLE_API_KEY=...) or Streamlit secrets."
    )


# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis using AI")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Uploaded Medical Image",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)
                    
                    # Create AgnoImage object
                    agno_image = AgnoImage(filepath=temp_path)
                    
                    # Run analysis
                    response: RunOutput = medical_agent.run(query, images=[agno_image])
                    st.markdown("### 📋 Analysis Results")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Note: This analysis is generated by AI and should be reviewed by "
                        "a qualified healthcare professional for accurate results."
                    )
                except Exception as e:
                    st.error(f"Analysis error: {e}")
else:
    st.info("👆 Please upload a medical image to begin analysis")
