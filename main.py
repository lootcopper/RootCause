from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CropDiseaseModel, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)

# Update paths and model loading (keeping your existing configuration)
train_dir = '/Users/pranay/softwareDEV/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
num_classes = 19
model = CropDiseaseModel(num_classes=num_classes)

model_path = '/Users/pranay/softwareDEV/crop_disease_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    model.eval()
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")

class_labels = sorted(os.listdir(train_dir))

# Function to retrieve disease insights from the custom dataset
def get_disease_insights(disease_name, confidence):
    """
    Get detailed insights about the disease using a custom JSON dataset
    """
    try:
        # Load the dataset
        with open("disease_data.json", "r") as file:
            data = json.load(file)
        
        # Get the disease information
        disease_info = data.get(disease_name, None)
        
        if disease_info:
            return {
                "description": disease_info["description"],
                "symptoms": disease_info["symptoms"],
                "treatment": disease_info["treatment"],
                "prevention": disease_info["prevention"]
            }
        else:
            return {
                "description": "Disease not found in the dataset.",
                "symptoms": ["Information unavailable"],
                "treatment": ["Information unavailable"],
                "prevention": ["Information unavailable"]
            }

    except Exception as e:
        print(f"Error getting insights: {str(e)}")
        return {
            "description": "Unable to fetch detailed insights at this time.",
            "symptoms": ["Information unavailable"],
            "treatment": ["Information unavailable"],
            "prevention": ["Information unavailable"]
        }

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    image_filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    file.save(file_path)

    # Perform prediction
    image = Image.open(file_path)
    processed_image = preprocess_image(image)
    
    with torch.no_grad():
        output = model(processed_image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
        confidence = confidence[predicted.item()].item()

    # Get detailed insights from the custom dataset
    insights = get_disease_insights(predicted_class, round(confidence, 2))

    # Return the results as JSON
    return jsonify({
        'image_url': f'/static/uploads/{image_filename}',
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'insights': insights
    })

if __name__ == '__main__':
    app.run(debug=True)
