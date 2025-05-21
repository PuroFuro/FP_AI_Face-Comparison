import streamlit as st
import os
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Face Comparison",
    page_icon="👤",
    layout="wide"
)

st.title("Face Similarity Comparison")

def load_image(image_file):
    if image_file is not None:
        img = Image.open(image_file)
        return img
    return None

def capture_image():
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        return img_file_buffer
    return None

def compare_faces(img1_path, img2_path):
    try:
        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"Cannot find image at {img1_path}")
        if not os.path.exists(img2_path):
            raise FileNotFoundError(f"Cannot find image at {img2_path}")
            
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("One or both images cannot be read properly")
        
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        faces1 = face_detector.detectMultiScale(gray_img1, 1.1, 4)
        faces2 = face_detector.detectMultiScale(gray_img2, 1.1, 4)
        
        if len(faces1) == 0:
            raise ValueError("No face detected in the first image. Please upload a clear face image.")
        if len(faces2) == 0:
            raise ValueError("No face detected in the second image. Please upload a clear face image.")
            
        result = DeepFace.verify(
            img1_path=img1_path, 
            img2_path=img2_path, 
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False,
            detector_backend=detector_backend
        )
        return result
    except FileNotFoundError as e:
        st.error(f"File error: {e}")
        return None
    except ValueError as e:
        st.error(f"Image error: {e}")
        return None
    except Exception as e:
        st.error(f"Error during face comparison: {str(e)}")
        st.info("Try with different images or ensure faces are clearly visible in both photos.")
        return None

def display_results(comparison_result):
    if comparison_result:
        with result_container:
            st.subheader("Comparison Results")
            verified = comparison_result["verified"]
            distance = comparison_result["distance"]
            threshold = comparison_result["threshold"]
            model = comparison_result["model"]
            
            similarity = max(0, 100 - (distance * 100))
            
            # Custom HTML for the similarity bar
            similarity_color = f"rgb({int(255 * (1 - similarity/100))}, {int(255 * similarity/100)}, 0)"
            similarity_bar_html = f"""
                <div style="margin-bottom: 20px;">
                    <div style="width: 100%; background-color: #f0f0f0; height: 30px; border-radius: 15px; overflow: hidden;">
                        <div style="width: {similarity}%; background-color: {similarity_color}; height: 100%; 
                             transition: width 0.5s ease-out;">
                        </div>
                    </div>
                    <div style="text-align: center; font-size: 1.5em; margin-top: 10px;">
                        Similarity: {similarity:.1f}%
                    </div>
                </div>
            """
            st.markdown(similarity_bar_html, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Distance", f"{distance:.4f}")
                st.metric("Threshold", f"{threshold:.4f}")
            
            with col2:
                match_text = "Match" if verified else "No Match"
                match_color = "green" if verified else "red"
                st.markdown(
                    f"""
                    <div style="height: 100%; display: flex; align-items: center; justify-content: center;">
                        <h1 style="color: {match_color}; font-size: 3em; text-align: center;">
                            {match_text}
                        </h1>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <span style="background-color: #f0f2f6; padding: 10px; border-radius: 10px;">
                    <strong>Model used</strong>: {model}, 
                    <strong>Face Detector</strong>: {detector_backend}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **How to interpret:**
            - Higher similarity percentage indicates closer match
            - Green bar indicates higher similarity, red indicates lower
            - Distance below threshold is considered a match
            """)

# Sidebar for navigation
with st.sidebar:
    st.header("Face Comparison")
    st.markdown("A tool to check similarity between two face images")
    
    comparison_mode = st.radio(
        "Comparison Mode",
        ["Image to Image", "Webcam to Image"],
        key="comparison_mode"
    )
    
    st.markdown("---")
    st.subheader("Model Settings")
    
    model_name = st.selectbox(
        "Face Recognition Model",
        ["VGG-Face", "Facenet512", "ArcFace", "SFace"],
        help="Select the model to use for face recognition. Different models may perform better in different scenarios."
    )
    
    distance_metric = st.selectbox(
        "Distance Metric",
        ["cosine", "euclidean", "euclidean_l2"],
        help="Method used to calculate the similarity between faces. Lower distance means higher similarity."
    )
    
    detector_backend = st.selectbox(
        "Face Detector",
        ["opencv", "ssd", "mtcnn", "retinaface"],
        help="Method used to detect faces in images. Choose based on accuracy vs speed trade-off."
    )

# Main content based on mode selection
if comparison_mode == "Image to Image":
    st.markdown("Upload two face images to check their similarity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Image")
        image_file1 = st.file_uploader("Upload first face image", type=["jpg", "jpeg", "png"], key="img1")
        if image_file1 is not None:
            image1 = load_image(image_file1)
            st.image(image1, caption="Image 1", use_container_width=True)

    with col2:
        st.subheader("Second Image")
        image_file2 = st.file_uploader("Upload second face image", type=["jpg", "jpeg", "png"], key="img2")
        if image_file2 is not None:
            image2 = load_image(image_file2)
            st.image(image2, caption="Image 2", use_container_width=True)

    # Create result container for image to image mode
    result_container = st.container()
    
    if st.button("Compare Faces", type="primary"):
        if image_file1 is not None and image_file2 is not None:
            with st.spinner("Comparing faces..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
                    tmp1.write(image_file1.getvalue())
                    img1_path = tmp1.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                    tmp2.write(image_file2.getvalue())
                    img2_path = tmp2.name
                    
                comparison_result = compare_faces(img1_path, img2_path)
                os.unlink(img1_path)
                os.unlink(img2_path)
                display_results(comparison_result)
        else:
            st.warning("Please upload both images first.")

else:  # Webcam to Image mode
    st.markdown("Capture a photo with your webcam and compare it with an uploaded image.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Webcam Capture")
        webcam_image = capture_image()
        
    with col2:
        st.subheader("Reference Image")
        reference_image = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"], key="ref_img")
        if reference_image is not None:
            image_ref = load_image(reference_image)
            st.image(image_ref, caption="Reference Image", use_container_width=True)
    
    # Create result container for webcam mode
    result_container = st.container()
    
    if st.button("Compare with Reference Image", type="primary"):
        if webcam_image is not None and reference_image is not None:
            with st.spinner("Comparing faces..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
                    tmp1.write(webcam_image.getvalue())
                    webcam_path = tmp1.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                    tmp2.write(reference_image.getvalue())
                    ref_path = tmp2.name
                    
                comparison_result = compare_faces(webcam_path, ref_path)
                os.unlink(webcam_path)
                os.unlink(ref_path)
                display_results(comparison_result)
        else:
            st.warning("Please capture a photo and upload a reference image first.")

with st.expander("About this app"):
    st.write("""
    This application uses DeepFace library to compare two face images and determine their similarity.
    
    **Features:**
    - Multiple face recognition models:
        - VGG-Face: Traditional and reliable
        - Facenet512: Google's deep learning model
        - ArcFace: State-of-the-art accuracy
        - SFace: Lightweight and fast
    
    - Multiple distance metrics:
        - Cosine: Angular difference between face features
        - Euclidean: Direct distance between face features
        - Euclidean L2: Normalized euclidean distance
    
    - Multiple face detectors:
        - OpenCV: Fast but basic
        - SSD: Single Shot Detector, good for multiple faces
        - MTCNN: Multi-task Cascaded CNN, very accurate
        - RetinaFace: State-of-the-art accuracy
    
    - Multiple comparison modes: 
        - Image to Image
        - Webcam to Image
    
    For best results:
    1. Upload clear face images or use good lighting with webcam
    2. Try different models if one doesn't work well
    3. Experiment with different distance metrics for better accuracy
    """)