from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize the VGGFace model and MTCNN detector
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()

# Load and detect face in the sample image
sample_img = cv2.imread('sample/Pritam.jpg')
results = detector.detect_faces(sample_img)

# Extract bounding box coordinates (ensure non-negative values)
x, y, width, height = results[0]['box']
x, y = max(0, x), max(0, y)
face = sample_img[y:y+height, x:x+width]

# Convert BGR to RGB and resize the face to 224x224
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
image = Image.fromarray(face).resize((224, 224))

# Convert the image to a numpy array and preprocess it
face_array = np.asarray(image).astype('float32')
preprocessed_img = preprocess_input(np.expand_dims(face_array, axis=0))

# Extract features using the VGGFace model
result = model.predict(preprocessed_img).flatten()

# Compute cosine similarity in a vectorized manner
similarity = cosine_similarity(result.reshape(1, -1), feature_list).flatten()

# Find the index of the best match
index_pos = np.argmax(similarity)
best_match_filename = filenames[index_pos]

# Display the best match image
temp_img = cv2.imread(best_match_filename)
if temp_img is not None:
    cv2.imshow('Best Match', temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()