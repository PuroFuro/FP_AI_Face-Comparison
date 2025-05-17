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
st.markdown("Upload two face images to check their similarity.")

col1, col2 = st.columns(2)

def load_image(image_file):
    if image_file is not None:
        img = Image.open(image_file)
        return img
    return None

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

result_container = st.container()

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
            model_name="VGG-Face", 
            distance_metric="cosine",
            enforce_detection=False,  
            detector_backend="opencv"  
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

            if comparison_result:
                with result_container:
                    st.subheader("Comparison Results")

                    verified = comparison_result["verified"]
                    distance = comparison_result["distance"]
                    threshold = comparison_result["threshold"]
                    model = comparison_result["model"]
                    
                    similarity = max(0, 100 - (distance * 100))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Similarity", f"{similarity:.2f}%")
                        st.metric("Distance", f"{distance:.4f}")
                    
                    with col2:
                        st.metric("Threshold", f"{threshold:.4f}")
                        match_text = "Match" if verified else "No Match"
                        match_color = "green" if verified else "red"
                        st.markdown(f"<h3 style='color: {match_color};'>{match_text}</h3>", unsafe_allow_html=True)
                        
                    st.markdown(f"**Model used**: {model}")
                    
                    st.info("""
                    **How to interpret:**
                    - Lower distance means higher similarity
                    - If distance is below threshold, images are considered a match
                    """)
    else:
        st.warning("Please upload both images first.")

with st.expander("About this app"):
    st.write("""
    This application uses DeepFace library to compare two face images and determine their similarity.
    
    **Features:**
    - Face verification using VGG-Face model
    - Cosine similarity metric for comparison
    - Similarity percentage calculation
    
    Upload two clear face images for the best results.
    """)

with st.sidebar:
    st.header("Face Comparison")
    st.markdown("A tool to check similarity between two face images")