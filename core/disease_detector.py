import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PlantDiseaseCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DiseaseDetector:
    def __init__(self, model_path=None):
        self.model = PlantDiseaseCNN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.disease_classes = [
            'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
            'leaf_mold', 'septoria_leaf_spot', 'spider_mites', 'target_spot',
            'yellow_leaf_curl', 'mosaic_virus'
        ]
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image).unsqueeze(0)
    
    def detect_diseases(self, image):
        processed_image = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        disease_name = self.disease_classes[predicted_class]
        
        severity = self.estimate_severity(image, disease_name)
        
        return {
            'disease': disease_name,
            'confidence': confidence,
            'severity': severity,
            'recommendation': self.get_treatment_recommendation(disease_name, severity)
        }
    
    def estimate_severity(self, image, disease_name):
        if disease_name == 'healthy':
            return 'none'
        
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        total_pixels = image.shape[0] * image.shape[1]
        affected_pixels = np.sum(brown_mask > 0) + np.sum(yellow_mask > 0)
        healthy_pixels = np.sum(green_mask > 0)
        
        if healthy_pixels == 0:
            severity_ratio = 1.0
        else:
            severity_ratio = affected_pixels / (affected_pixels + healthy_pixels)
        
        if severity_ratio < 0.1:
            return 'low'
        elif severity_ratio < 0.3:
            return 'medium'
        else:
            return 'high'
    
    def get_treatment_recommendation(self, disease_name, severity):
        treatments = {
            'bacterial_spot': {
                'low': 'Apply copper-based bactericide weekly',
                'medium': 'Apply copper-based bactericide twice weekly, remove affected leaves',
                'high': 'Apply systemic bactericide, remove severely affected plants'
            },
            'early_blight': {
                'low': 'Apply fungicide containing chlorothalonil',
                'medium': 'Apply fungicide weekly, improve air circulation',
                'high': 'Apply systemic fungicide, remove infected plants'
            },
            'late_blight': {
                'low': 'Apply copper fungicide preventatively',
                'medium': 'Apply systemic fungicide, remove affected leaves',
                'high': 'Emergency fungicide application, destroy infected plants'
            },
            'healthy': {
                'none': 'Continue current maintenance practices'
            }
        }
        
        return treatments.get(disease_name, {}).get(severity, 'Consult agricultural expert')