import gradio as gr
import requests

# API endpoint backend
API_URL = "http://127.0.0.1:8000/predict/"

# Gửi yêu cầu POST với ảnh đến API backend
def predict(image):
    response = requests.post(API_URL, files={"file": image})
    result = response.json()
    return result['label']

# Giao diện Gradio
iface = gr.Interface(fn=predict, inputs=gr.Image(type="file"), outputs="text", 
                     title="Cat vs Dog Classifier", 
                     description="Upload an image to classify it as either a Cat or a Dog.")

if __name__ == "__main__":
    iface.launch()
