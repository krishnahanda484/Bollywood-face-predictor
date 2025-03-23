import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Load filenames if they exist, otherwise collect them
if os.path.exists('filenames.pkl'):
    filenames = pickle.load(open('filenames.pkl', 'rb'))
else:
    base_dir = 'Bollywood_celeb_face_localized'
    filenames = [
        os.path.join(subdir_path, actor, file)
        for subdir in os.listdir(base_dir)
        for subdir_path in [os.path.join(base_dir, subdir)]
        if os.path.isdir(subdir_path)
        for actor in os.listdir(subdir_path)
        for actor_path in [os.path.join(subdir_path, actor)]
        if os.path.isdir(actor_path)
        for file in os.listdir(actor_path)
        if os.path.isfile(os.path.join(actor_path, file))
    ]
    with open('filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)

# Load the VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Feature extractor function
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    preprocessed_img = preprocess_input(np.expand_dims(img_array, axis=0), version=2)
    return model.predict(preprocessed_img).flatten()

# Extract features for all files
features = [feature_extractor(file, model) for file in tqdm(filenames, desc="Extracting features")]

# Save the features to a pickle file
pickle.dump(features, open('embedding.pkl', 'wb'))