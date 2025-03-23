#!pip install streamlit
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# Ensure the 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Initialize the MTCNN detector and VGGFace model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed features and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if not results:
        return None

    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y + height, x:x + width]

    # Extract features
    image = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    preprocessed_img = preprocess_input(np.expand_dims(face_array, axis=0))
    return model.predict(preprocessed_img).flatten()

def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0] for i in range(len(feature_list))]
    return sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)[0][0]

# Streamlit app
st.title('Which Bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)

        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

            col1, col2 = st.columns(2)
            with col1:
                st.header('Your uploaded image')
                st.image(display_image)
            with col2:
                st.header(f"Seems like {predicted_actor}")
                st.image(filenames[index_pos], width=300)
        else:
            st.error("No face detected in the uploaded image!")