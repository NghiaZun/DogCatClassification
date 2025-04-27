app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Lazy loading model
model = None

# ... transforms ...

def load_model():
    global model
    if model is None:
        print("Loading model...")
        repo_id = "Gumangusi/CatDogClassifier"
        filename = "catdog_final.pth"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        model = CatDogClassifier(backbone_name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained_backbone=False)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()

load_model()

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    print("Received image:", img_file.filename)

    try:
        print("Processing image...")
        img = Image.open(img_file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.sigmoid(output).item()
            label = "Dog" if pred > 0.5 else "Cat"

        print("Prediction:", label)
        return jsonify({'label': label})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500
