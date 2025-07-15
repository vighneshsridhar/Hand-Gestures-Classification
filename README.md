Hand Gesture Recognition with CNN & Real-Time Video
This project is a hand gesture recognition system that uses:

A Convolutional Neural Network (CNN) to classify gestures from 3D hand landmarks.

Real-time video feed using your webcam and MediaPipe to extract hand landmarks.

A trained model (.h5) and label encoder (.pkl) to recognize gestures live.

The system can detect and classify 21-point 3D hand landmarks into custom gesture classes that you train.

Features
Train a CNN on normalized 21×3 landmarks (x, y, z) of the hand.
Normalize landmarks (relative to wrist & scaled) to improve generalization.
Save and load models: .h5 (CNN) and .pkl (LabelEncoder).
Real-time gesture prediction & display on webcam feed.
Configurable confidence threshold and camera index.

Project Structure
bash
Copy
Edit
.
├── exports/
│   ├── cnn_model.h5         # Trained CNN model
│   └── cnn_encoder.pkl      # Trained LabelEncoder
├── dataset/
│   ├── train.csv            # Training data (landmarks + labels)
│   └── test.csv             # Optional test data
├── gesture_recognizer.py    # Real-time recognition script
├── cnn_training.py          # CNN training script
├── utils.py                 # Utility functions
└── README.md                # This file
🧪 Requirements
Python 3.8+

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:
Copy
Edit
opencv-python
mediapipe
tensorflow
scikit-learn
numpy
How to Train
1️ Prepare your dataset (train.csv) — each row should be:

css
Copy
Edit
x1,y1,z1,x2,y2,z2,...,x21,y21,z21,label
2️ Run the training script:

bash
Copy
Edit
python cnn_training.py --data dataset/train.csv --epochs 20 --batch_size 32
This will output:

Trained CNN: exports/cnn_model.h5

Label encoder: exports/cnn_encoder.pkl

Real-Time Gesture Recognition
Make sure you have the trained model & encoder (.h5, .pkl) in the exports/ folder.

Run the real-time recognizer:

bash
Copy
Edit
python gesture_recognizer.py \
    --model exports/cnn_model.h5 \
    --encoder exports/cnn_encoder.pkl \
    --camera 0 \
    --threshold 0.6
Options:

Argument	Description
--model	Path to the .h5 CNN model file
--encoder	Path to the .pkl LabelEncoder
--camera	Camera index (default: 0)
--threshold	Confidence threshold (default: 0.5)

Press q to quit the video stream.

Notes
  The landmarks are normalized relative to the wrist and scaled to fit into [-1, 1].
  Make sure the gestures you train and test are consistent in format and scale.
  Works fully offline once the model is trained.

Example Output


Contributions
Contributions, issues, and feature requests are welcome! Please open an issue or pull request.

License
This project is licensed under the MIT License — see the LICENSE file for details.
