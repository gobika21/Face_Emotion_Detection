
# Face Emotion Detection App

This is a **Face Emotion Detection** app built using **Streamlit** and **TensorFlow**. The application allows users to upload an image, and it predicts the emotion displayed on the face using a pre-trained Convolutional Neural Network (CNN) model.

## Features
- Allows image uploads for emotion detection.
- Detects emotions such as angry, disgust, fear, happy, sad, surprise, and neutral.
- Simple and user-friendly interface powered by Streamlit.

## Model
This app uses a pre-trained model saved in the Keras format (`ED.keras`). The model is trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset, which contains labeled images for different emotions.

## Prerequisites

- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/install) for model building and inference
- [Streamlit](https://docs.streamlit.io/) for creating the web interface

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gobika21/Face_Emotion_Detection.git
   cd Face_Emotion_Detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure your model is saved as `your_model_path.keras` (or adjust the path in the code) and place it in the root directory of the project.

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload an image containing a face, and the app will display the predicted emotion.

## File Structure

- `app.py`: Main application code that handles the Streamlit interface, image upload, and emotion prediction.
- `ED.keras`: Pre-trained model file. Ensure it is in the correct format and path.
- `requirements.txt`: Contains the dependencies required to run this project.

## Example

Upload an image (in `.jpg`, `.jpeg`, or `.png` format), and the app will predict the emotion. Below is a sample workflow:

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an image**: Choose a facial image from your device.

3. **View prediction**: The app will display the uploaded image and predict the emotion, showing results like "Happy," "Sad," or "Surprise."

## Requirements

Create a `requirements.txt` file with the following contents:

```text
streamlit
tensorflow
Pillow
numpy
```

You can install these dependencies by running:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- **Incorrect Predictions**: If predictions are inaccurate, ensure the model is trained well and has enough balanced data.
- **Model Path Issue**: Make sure `ED.keras` points to the correct location of your model file in `app.py`.
