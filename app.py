import gradio as gr
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("rf_mnist_model.joblib")

# Fixed predict function
def predict_digit(img_dict):
    try:
        img = img_dict["layers"][0]
        img = Image.fromarray(img).convert("L").resize((28, 28))
        img = np.array(img)
        img = 255 - img
        img = img / 255.0
        img = img.reshape(1, -1)
        prediction = model.predict(img)[0]
        return f"Predicted Digit: {prediction}"
    except Exception as e:
        return f"Error: {str(e)}"

# UI
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(),
    title="MNIST Digit Recognizer",
    description="Draw a digit (0–9) and the model will guess it!"
)

interface.launch(share=True)
