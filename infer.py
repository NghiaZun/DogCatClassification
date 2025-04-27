import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model_arch import CatDogClassifier
from huggingface_hub import hf_hub_download

# Tải mô hình từ Hugging Face
device = 'cuda' if torch.cuda.is_available() else 'cpu'
repo_id = "Gumangusi/CatDogClassifier"  # Thay bằng tên repo của bạn trên Hugging Face
filename = "catdog_final.pth"  # Tên file mô hình đã upload

# Tải file mô hình từ Hugging Face
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load lại model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CatDogClassifier(backbone_name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained_backbone=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Chuyển đổi ảnh trước khi đưa vào mô hình
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    return image

# Hàm dự đoán cho 1 bức ảnh
def predict(model, image_tensor):
    with torch.no_grad():  # Không tính gradient để tiết kiệm bộ nhớ
        output = model(image_tensor)
        preds = torch.sigmoid(output)  
        predicted = (preds > 0.5).int()  
    return predicted.item()


# Kiểm tra mô hình với một bức ảnh
def infer(image_path, model):
    image_tensor = preprocess_image(image_path).to(device)
    label = predict(model, image_tensor)
    
    # Mapping index -> class name
    idx_to_class = {0: 'Cat', 1: 'Dog'}
    
    pred_label = idx_to_class[label]
    
    # Hiển thị kết quả
    print(f'Predicted: {pred_label}')
    
    # Vẽ ảnh
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Predicted: {pred_label}')
    plt.show()

if __name__ == "__main__":
    image_path = 'img/meo_2.jpg'  
    infer(image_path, model)
