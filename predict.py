import numpy as np
from keras.preprocessing import image
from keras.models import load_model

def predict_image(filepath):
    # Load the model
    model = load_model('model.h5')  # Replace 'your_model.h5' with the path to your trained model
    
    # Preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    # Predict the image
    prediction = model.predict(img)
    
    # Interpret the prediction
    if prediction[0] >= 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"
    
    return result
