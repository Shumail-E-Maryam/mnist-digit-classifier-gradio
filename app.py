import gradio as gr
import numpy as np
import joblib
from PIL import Image

# Load your trained model (already trained and saved)
model = joblib.load("rf_mnist_model.joblib")

# Prediction logic
def predict_digit(image):
    # Convert sketchpad image to grayscale 28x28
    image = Image.fromarray(image).convert("L").resize((28, 28))
    image = np.array(image).reshape(1, -1)
    image = 255 - image  # Invert colors
    return int(model.predict(image)[0])

# Gradio Interface using Sketchpad (Gradio 5.34.2 syntax)
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(),  # ✅ No shape/brush
    outputs=gr.Label(),
    title="MNIST Digit Recognizer",
    description="Draw a digit (0–9) and the model will guess it!"
)

# Launch the app
interface.launch(share=True)
