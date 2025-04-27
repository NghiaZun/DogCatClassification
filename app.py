from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from model_arch import CatDogClassifier

app = Flask(__name__)

# Load pre-trained model (example with ResNet18)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
repo_id = "Gumangusi/CatDogClassifier"  # Thay bằng repo của bạn
filename = "catdog_final.pth"
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load mô hình
model = CatDogClassifier(backbone_name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained_backbone=False)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()  # Đặt mô hình ở chế độ đánh giá
logger.info("Model loaded successfully")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img = Image.open(img_file.stream)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output, 1)
    label = 'Cat' if predicted.item() == 0 else 'Dog'
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
