import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "muffins"]  # will use this to convert prediction num to string value

def prepare(filepath):
    IMG_SIZE = 80  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("trained.model")

prediction = model.predict([prepare('image 4.jpg')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])

#print("hi")

