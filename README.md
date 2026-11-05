# Sign Language Greeting Detection  

Real-time Sign Language Recognition using Flask, MediaPipe, and PyTorch



# Overview


This project detects common greeting sign-language gestures in real-time using a webcam, and runs in a browser as a website. It uses MediaPipe for hand landmark extraction and a PyTorch classifier for predicting which greeting is shown. The frontend captures frames and posts them to the Flask backend for inference.

# Tech Stack

- Programming Language:** Python 3
- Backend Framework:** Flask
- Computer Vision:** MediaPipe (Hand Landmarks)
- Deep Learning:** PyTorch
- Frontend:** HTML, CSS, JavaScript
- Database:** SQLite
- Model Types:** MLP / Transformer (optional)
- Tools:** NumPy, OpenCV


# Features
- Real-time detection (~200-400ms per frame depending on hardware)
- Light-weight pipeline (landmarks + MLP) for fast inference on CPU
- Modular: swap MLP for ViT (Hugging Face transformers) if higher accuracy is needed
- Simple dataset collection script included to create `.npy` landmark samples
- History tracking of detected gestures with timestamps
- Statistics dashboard for gesture frequency analysis
- User-friendly web interface with responsive design

# How it works (pipeline)
1. Browser captures webcam frames and sends base64 JPEG to `/predict`.
2. Backend decodes image → uses MediaPipe Hands to extract 21 hand landmarks.
3. Landmarks are flattened into a 63-d vector and passed to the PyTorch classifier.
4. JSON returned with predicted label & confidence → frontend displays it.
5. Detected gestures are stored in a SQLite database for history tracking.

# Project Structure
```
├── app.py                  # Main Flask application
├── capture_to_npy.py       # Script to capture training data
├── create_dummy_model.py   # Creates a placeholder model for testing
├── dataset/                # Training data storage
├── instance/               # SQLite database location
├── model/                  # Neural network model files
│   ├── model.py            # Model architecture definition
│   └── train_transformer.py # Training script for transformer model
├── static/                 # Static assets (JS, CSS)
├── templates/              # HTML templates
└── requirements.txt        # Project dependencies
```

# Prerequisites
- Python 3.8+
- Webcam
- Modern web browser with JavaScript enabled

# Installation
1. Clone the repository:
   ```
   git clone https://github.com/Abhichanda2003/Sign-Language-AI.git
   cd Sign-Language-AI
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

# Training Your Own Model
- Capture examples with `capture_to_npy.py` (one label per run).
- Organize data as `dataset/landmarks.npy` and `dataset/labels.npy`.
- Run `python model/train_transformer.py` to train and save `model/checkpoint.pth`.

# Running the Application
1. Ensure you have a trained model at `model/checkpoint.pth` or run:
   ```
   python create_dummy_model.py
   ```
   to create a placeholder model for testing.

2. Start the Flask server:
   ```
   python app.py
   ```

3. Open `http://localhost:5000` in your browser.

4. Grant camera permissions when prompted.

# Features in Detail
- **Real-time Detection**: View sign language detection results as you sign
- **Gesture History**: Review past detected gestures with timestamps
- **Statistics**: Analyze frequency and patterns of detected gestures
- **About Page**: Learn more about sign language and the project

# Future Improvements
- For better accuracy use data augmentation or switch to an image model (ViT) fine-tuned on your dataset
- Use WebSockets (Flask-SocketIO) for lower latency streaming
- Add multi-hand handling, smoothing over time (temporal averaging), and confidence thresholding
- Implement user accounts for personalized history tracking
- Add mobile device support with responsive design optimizations
- Expand the gesture vocabulary beyond basic greetings

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.



