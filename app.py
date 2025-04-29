import streamlit as st
import os
import io
import time
from PIL import Image
from blurblend import BlurBlend

# Page configuration
st.set_page_config(
    page_title="BlurBlend - AI Background Blur",
    page_icon="🖼️",
    layout="wide"
)

# Header
st.title("BlurBlend 🖼️")
st.markdown("### AI-powered background blur for your images")

# Sidebar controls
st.sidebar.header("Settings")

# Model selection
model_options = {
    "Segmentation Model (Default)": "facebook/mask2former-swin-base-coco-panoptic",
    "Smaller Model": "nvidia/segformer-b0-finetuned-ade-512-512",
    "Person Detection": "facebook/detr-resnet-50-panoptic",
    "Person Detection +": "mattmdjaga/segformer_b2_clothes"
}

selected_model = st.sidebar.selectbox(
    "Choose segmentation model", 
    list(model_options.keys())
)

# Blur settings
blur_radius = st.sidebar.slider(
    "Blur Intensity", 
    min_value=1, 
    max_value=50, 
    value=15,
    help="Higher values create a stronger blur effect"
)

# Initialize model with caching to prevent reloading on each interaction
@st.cache_resource
def load_blurblend_model(model_name):
    """Load BlurBlend model with fallback options and error handling"""
    model_path = model_options[model_name]
    
    with st.spinner(f"Loading model {model_name}..."):
        try:
            return BlurBlend(model_path=model_path)
        except Exception as e:
            st.warning(f"Failed to load model {model_name}.")
            
            # Try a fallback if the primary model fails
            if model_name != "Smaller Model":
                st.info("Attempting to load smaller alternative model...")
                try:
                    fallback_model = BlurBlend(model_path=model_options["Smaller Model"])
                    st.success("Successfully loaded fallback model.")
                    return fallback_model
                except Exception as fallback_e:
                    st.error("Failed to load fallback model.")
            
            # Return None if all attempts fail
            return None

# Main content area
st.subheader("Upload & Process Image")

# Use columns with equal width for desktop view
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

with col2:
    if uploaded_file is not None:
        # Initialize the model based on selection with error handling
        try:
            model = load_blurblend_model(selected_model)
            
            if model is None:
                st.error("Unable to load any model. Please try again later or choose a different model.")
            else:
                st.write("") 
                st.write("") 
                st.write("") 
                st.write("") 

                # Process button 
                if st.button("Apply Background Blur", use_container_width=True):
                    with st.spinner("Processing image..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = "temp_upload.jpg"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process image
                            result_img = model.process_image(
                                image_path=temp_path,
                                blur_radius=blur_radius
                            )
                            # Create download button
                            buf = io.BytesIO()
                            result_img.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            # Display result first so user can see it immediately
                            st.image(result_img, caption="Blurred Background", use_container_width=True)
                            
                            # Then add download button
                            st.download_button(
                                label="Download Result",
                                data=byte_im,
                                file_name="blurblend_result.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                            # Suggest solutions
                            st.info("Try using a different model or uploading a different image.")
        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")
            st.info("Please refresh the page and try again.")
    else:
        st.info("Upload an image to see the result here")


st.markdown("---")
st.markdown("""
## How it works
1. Upload an image containing people or objects
2. The AI model identifies foreground subjects 
3. BlurBlend applies a blur effect to the background while keeping the foreground sharp

For best results, use images with clear subjects against a distinct background.
""")

# Model information
st.markdown("""
## About the models
- **Segmentation Model (Default)**: Best overall quality for various subjects
- **Smaller Model**: Faster but less accurate, good for simple images
- **Person Detection**: Specialized in detecting people
- **Person Detection +**: Better for clothing and fashion images
""")

# Troubleshooting section
with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common issues:
    - **Model loading fails**: Try selecting the "Smaller Model" option
    - **Poor segmentation**: Try a different model that might work better for your image
    - **Processing takes too long**: Choose the "Smaller Model" for faster results
    - **App crashes**: Refresh the page and try again with a smaller image
    """)

# Footer
st.markdown("---")
st.caption("BlurBlend • AI-Powered Background Blur")

# Social media links row
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kavishannip)")
with col2:
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kavishan-nipun-876930222/)")

st.caption("© developed by Kavishan Nipun")