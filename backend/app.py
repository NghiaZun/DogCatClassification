from huggingface_hub import hf_hub_download
import torch
import timm
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from torchvision import transforms
from model_arch import load_model

# Khởi tạo FastAPI app
app = FastAPI()

# Tải mô hình từ Hugging Face
device = 'cuda' if torch.cuda.is_available() else 'cpu'
repo_id = "Gumangusi/CatDogClassifier"  # Thay bằng tên repo của bạn trên Hugging Face
filename = "catdog_final.pth"  # Tên file mô hình đã upload

# Tải file mô hình từ Hugging Face
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load mô hình
model = CatDogClassifier(backbone_name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained_backbone=False)
model.load_state_dict(torch.load('catdog_final.pth', map_location=device))
model = model.to(device)

# Hàm chuẩn hóa ảnh và dự đoán
def predict(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output).item()
    
    return "Dog" if pred > 0.5 else "Cat"

# API nhận ảnh và trả về dự đoán
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    label = predict(img)
    return {"label": label}