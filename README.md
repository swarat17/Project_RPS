# Stone Paper Scissors Classifier

## Overview

This repository contains code for a deep learning project where a Convolutional Neural Network (CNN) was constructed from scratch to classify hand gestures for the Stone Paper Scissors game. The model was trained on an open-source dataset and achieved an impressive **96% accuracy**. Additionally, the project includes a real-time classification interface implemented using OpenCV, allowing users to classify hand gestures directly from webcam footage.

---

## Features

- **CNN from Scratch**: A custom-designed Convolutional Neural Network architecture optimized for gesture classification.
- **High Accuracy**: Achieved a robust classification accuracy of 96% on the test dataset.
- **Real-Time Classification**: Integrated with OpenCV to enable real-time predictions via webcam.
- **Script-Based Dataset Setup**: A shell script (`get_data.sh`) simplifies downloading and organizing the dataset.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/swarat17/Project_RPS.git
   cd Project_RPS-master
   ```

2. **Install dependencies**:
   Ensure you have Python installed. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the dataset**:
   Run the included script to download and organize the dataset:
   ```bash
   sh get_data.sh
   ```
   This will populate the `rps` folder with subfolders (`rock`, `paper`, and `scissors`) for training data and `rps-test-set` for testing data.

---

## Usage

### Training the Model
Train the CNN using the provided dataset:
```bash
python train.py
```
This script will train the model and save it as `rps.h5`.

### Testing the Model
Test the trained model on a set of test images:
```bash
python test_image.py
```

### Real-Time Classification
Launch the real-time classification interface using your webcam:
```bash
python test_live.py
```
Ensure your webcam is functional, and the interface will display live footage with predictions for hand gestures.

---

## Directory Structure

```
stone-paper-scissors-classifier/
├── rps/                  # Training dataset (subfolders: rock, paper, scissors)
├── rps-test-set/         # Testing dataset (subfolders: rock, paper, scissors)
├── get_data.sh           # Script to download and organize the dataset
├── requirements.txt      # Python dependencies
├── rps.h5                # Saved model
├── train.py              # Script for training the model
├── test_image.py         # Script for testing the model with test images
├── test_live.py          # Script for real-time webcam classification
```

---

## Key Files

- **`train.py`**: Trains the CNN and saves the trained model as `rps.h5`.
- **`test_image.py`**: Tests the trained model on images from the test set.
- **`test_live.py`**: Uses OpenCV to perform real-time hand gesture classification using the webcam.
- **`get_data.sh`**: Shell script for downloading and organizing the dataset.

---

## Results

- **Model Accuracy**: Achieved 96% accuracy on the test dataset.
- **Real-Time Performance**: The OpenCV interface classifies gestures with minimal latency, ensuring a seamless experience.

---

## Technologies Used

- **Frameworks**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Programming Language**: Python
- **Dataset**: Open-source dataset for Stone Paper Scissors gestures

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for new features, bug fixes, or improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Thanks to the creators of the open-source dataset.
- Inspired by the classic game of Stone Paper Scissors and the potential of real-time AI applications.

--- 
