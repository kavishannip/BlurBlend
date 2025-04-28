import streamlit as st
import os
import io
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
    model_path = model_options[model_name]
    with st.spinner(f"Loading model {model_path}..."):
        return BlurBlend(model_path=model_path)

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
        # Initialize the model based on selection
        model = load_blurblend_model(selected_model)

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
                    
                    st.download_button(
                        label="Download Result",
                        data=byte_im,
                        file_name="blurblend_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # Display result
                    st.image(result_img, caption="Blurred Background", use_container_width=True)
                    
                    
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.info("Upload an image to see the result here")

# Add information about the app
st.markdown("---")
st.markdown("""
## How it works
1. Upload an image containing people
2. The AI model identifies foreground objects
3. BlurBlend applies a blur effect to the background while keeping the foreground sharp

For best results, use images with clear subjects against a distinct background.
""")

# Footer
st.markdown("---")
st.caption("BlurBlend • AI-Powered Background Blur")
st.caption("© developed byKavishan Nipun")