from tensorflow import keras
import cv2
import numpy as np

REV_CLASS_MAP = {
    0: "paper",
    1: "rock",
    2: "scissors"
}


def mapper(val):
    return REV_CLASS_MAP[val]

model = keras.models.load_model("rps.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to place hand
    cv2.rectangle(frame, (50, 50), (250, 250), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[50:250, 50:250]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Prediction: " + user_move_name,
        (50, 50), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()