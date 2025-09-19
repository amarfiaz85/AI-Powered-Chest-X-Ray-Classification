import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Define classes
CLASSES = ["NORMAL", "PNEUMONIA"]

# Load TorchScript model
MODEL_PATH = "resnet101_trained.pt"
model = torch.jit.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Homepage
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Save uploaded file
    upload_path = os.path.join("static/uploads", file.filename)
    file.save(upload_path)

    # Preprocess
    image = Image.open(upload_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = CLASSES[predicted.item()]

    return render_template("index.html", prediction=label, img_path=upload_path)

# Run app
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
