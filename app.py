import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from huggingface_hub import hf_hub_download

# ------------------------------
# Config
# ------------------------------
MODEL_REPO = "amarfiaz85/resnet101-chest-xray"
MODEL_FILE = "resnet101_trained.pt"

# ------------------------------
# Download model from Hugging Face Hub
# ------------------------------
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model downloaded to: {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    # You might want to add fallback logic here

# ------------------------------
# Load model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Assuming 2 classes: Normal, Pneumonia
num_classes = 2
model = models.resnet101(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# ------------------------------
# Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ["Normal", "Pneumonia"]

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            try:
                img = Image.open(file).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_t)
                    _, predicted = torch.max(outputs, 1)
                    label = class_names[predicted.item()]

                return render_template("result.html", label=label)
            except Exception as e:
                return render_template("error.html", error=str(e))

    return render_template("index.html")

# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
