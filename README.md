# Emotion Recognition

This project implements a Convolutional Neural Network (CNN) for facial emotion recognition. It is designed to classify facial expressions into 7 distinct emotion categories: **angry, disgust, fear, happy, neutral, sad, and surprise**.

The model is built using TensorFlow and Keras, featuring a custom architecture (`RealTimeCNN`) optimized for efficiency.

## Model Architecture

The `RealTimeCNN` model (defined in `src/model.py`) is a custom Convolutional Neural Network tailored for 48x48 pixel input images.
- **Layers**: It consists of 4 stacked Convolutional blocks.
- **Components**: Each block includes Convolutional layers (`Conv2D`), Batch Normalization, Max Pooling, and Dropout for regularization.
- **Classification**: The top layers are fully connected (`Dense`) with Softmax activation to output probabilities for the 7 emotion classes.

## Installation

Ensure you have **Python 3.10+** installed.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YiannisParask/facial-emotion-recognition.git
    cd facial-emotion-recognition
    ```

2.  **Set up a Virtual Environment**:
    ```bash
    python -m venv .venv
    # or with uv
    uv venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    This project is configured with `pyproject.toml`.
    ```bash
    uv pip install -r pyproject.toml
    ```

## Dataset

The project assumes a dataset structure similar to FER-2013:
- **Image Size**: 48x48 pixels.
- **Classes**: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- **Location**: Place your data in `data/train` and `data/test`.

## Usage

### Training & Exploration
The core training logic and analysis can be found in the notebook:
```bash
jupyter notebook notebooks/emotion-image-recognition.ipynb
```
This notebook handles data loading, preprocessing, model training, and performance evaluation.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE)
file for more details.
