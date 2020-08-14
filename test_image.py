from tensorflow import keras
import cv2
import numpy as np
import sys

filepath = sys.argv[1]

REV_CLASS_MAP = {
    0: "paper",
    1: "rock",
    2: "scissors"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = keras.models.load_model("rps.h5")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (150, 150))

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))