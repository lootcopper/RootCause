from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(CropDiseaseModel, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

train_dir = '/Users/pranay/softwareDEV/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
class_labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_labels)

print(f"Detected {num_classes} classes: {class_labels}")

model = CropDiseaseModel(num_classes=num_classes)
model_path = '/Users/pranay/softwareDEV/best_crop_disease_model.pth' 
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")

def get_disease_insights(disease_name, confidence):
    try:
        with open("disease_data.json", "r") as file:
            data = json.load(file)
        
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
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

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

    image_filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    file.save(file_path)

    image = Image.open(file_path).convert('RGB')
    processed_image = preprocess_image(image).to(device)
    

    temperature = 0.5  
    with torch.no_grad():
        output = model(processed_image)
        probabilities = torch.nn.functional.softmax(output / temperature, dim=1) 
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_labels[predicted_idx.item()]
        confidence_value = confidence.item() * 100

    insights = get_disease_insights(predicted_class, round(confidence_value, 2))

    return jsonify({
        'image_url': f'/static/uploads/{image_filename}',
        'predicted_class': predicted_class,
        'confidence': round(confidence_value, 2),
        'insights': insights
    })

if __name__ == '__main__':
    app.run(debug=True)
