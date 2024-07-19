# EmoClassification: Image Emotion Classifier 😊😢

EmoClassification is a deep learning project that classifies images as either "happy" or "sad" using a pre-trained convolutional neural network. This tool allows users to quickly and accurately determine the emotional content of images, making it useful for a variety of applications in sentiment analysis, user experience research, and more.

## Features 🌟

- **Binary Emotion Classification**: Classify images as either "happy" or "sad"
- **Pre-trained Model**: Utilize a ready-to-use model for quick predictions
- **Fast Inference**: Optimized for quick prediction times

## Technologies Used 🛠️

- [TensorFlow](https://www.tensorflow.org/): For building and loading the deep learning model
- [Keras](https://keras.io/): High-level neural network API
- [NumPy](https://numpy.org/): For efficient numerical computations
- [Pillow](https://python-pillow.org/): For image processing tasks

## Installation 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EmoClassification.git
   cd EmoClassification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model:
   - Ensure you have the `happysadmodel.h5` file in the `models/` directory

3. Your can create your own model with even more data:
   - The training Notebook is provided  in the  repository


## Usage 🖥️

To use the pre-trained model for emotion classification, follow these steps:

1. Import the necessary libraries:
   ```python
   from tensorflow.keras.models import load_model
   import numpy as np
   from PIL import Image
   ```

2. Load the pre-trained model:
   ```python
   model = load_model('models/happysadmodel.h5')
   ```

3. Prepare your image:
   ```python
   def prepare_image(image_path, target_size=(224, 224)):
       img = Image.open(image_path)
       img = img.resize(target_size)
       img_array = np.array(img)
       img_array = np.expand_dims(img_array, axis=0)
       return img_array / 255.0  # Normalize the image
   ```

4. Make a prediction:
   ```python
   image_path = 'path/to/your/image.jpg'
   prepared_image = prepare_image(image_path)
   prediction = model.predict(prepared_image)
   
   emotion = "Happy" if prediction[0][0] > 0.5 else "Sad"
   confidence = prediction[0][0] if emotion == "Happy" else 1 - prediction[0][0]
   
   print(f"Predicted emotion: {emotion}")
   print(f"Confidence: {confidence:.2f}")
   ```



## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.
