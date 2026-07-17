import os
from PIL import Image as PILImage
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.agent import RunOutput
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# --------------------------------------------------------
# Load Google API Key from .env
# --------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "❌ GOOGLE_API_KEY not found.\n\n"
        "Create a .env file in your project folder with:\n\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )
    st.stop()

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------
with st.sidebar:
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )

    st.warning(
        "⚠ DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

# --------------------------------------------------------
# Initialize AI Agent
# --------------------------------------------------------
medical_agent = Agent(
    model=Gemini(
        id="gemini-3.1-flash-lite",
        api_key=GOOGLE_API_KEY,
    ),
    tools=[DuckDuckGoTools()],
    markdown=True,
)

# --------------------------------------------------------
# Medical Analysis Prompt
# --------------------------------------------------------
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.

Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal / Mild / Moderate / Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence
- Note any urgent or critical findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple language.
- Avoid medical jargon whenever possible.
- Include simple analogies if useful.
- Address common patient concerns.

### 5. Research Context
Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases.
- Search standard treatment protocols.
- Provide relevant medical links.
- Mention technological advances if applicable.
- Include 2–3 references.

Format your response using markdown with headings and bullet points.
"""

# --------------------------------------------------------
# UI
# --------------------------------------------------------
st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for AI-powered professional analysis.")

upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
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
                use_container_width=True,
            )

            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True,
            )

    with analysis_container:

        if analyze_button:

            with st.spinner("🔄 Analyzing image... Please wait."):

                try:

                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)

                    agno_image = AgnoImage(filepath=temp_path)

                    response: RunOutput = medical_agent.run(
                        query,
                        images=[agno_image],
                    )

                    st.markdown("## 📋 Analysis Results")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")

                    st.caption(
                        "This analysis is generated by AI and is intended for "
                        "educational purposes only. Always consult a qualified "
                        "healthcare professional for diagnosis and treatment."
                    )

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

else:
    st.info("👆 Please upload a medical image to begin analysis.")