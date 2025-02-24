# loading required libraries
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# model building
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# printing model summary
print(model.summary())

# function for extracting image features
def get_features(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# getting images path and name
path = (r"tshirt-data/") # this you can change as per your image path
filenames = []

for file in os.listdir(path):
    filenames.append(os.path.join(path, file))

# getting images features in a list
feature_list = []

for file in tqdm(filenames):
    feature_list.append(get_features(file, model))

# dumping filename and features in pickel file
pickle.dump(feature_list,open('features.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))





