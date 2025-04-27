from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model_arch import CatDogClassifier

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Lazy loading model
model = None
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    global model
    if model is None:
        repo_id = "Gumangusi/CatDogClassifier"
        filename = "catdog_final.pth"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        model = CatDogClassifier(backbone_name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained_backbone=False)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    load_model()

    img_file = request.files['image']
    print("Received image:", img_file.filename)  # Log để kiểm tra
    
    try:
        img = Image.open(img_file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.sigmoid(output).item()
            label = "Dog" if pred > 0.5 else "Cat"

        print("Prediction:", label)  # Log kết quả

        return jsonify({'label': label})
    except Exception as e:
        print(f"Error: {str(e)}")  # Log chi tiết lỗi
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=False)