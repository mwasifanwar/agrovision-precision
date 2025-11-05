import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

class SoilQualityCNN(nn.Module):
    def __init__(self):
        super(SoilQualityCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

class SoilAnalyzer:
    def __init__(self):
        self.soil_model = SoilQualityCNN()
        self.moisture_predictor = RandomForestRegressor()
        self.color_ranges = {
            'clay': {'lower': [0, 0, 50], 'upper': [20, 255, 150]},
            'sandy': {'lower': [20, 50, 150], 'upper': [40, 255, 255]},
            'loamy': {'lower': [10, 30, 100], 'upper': [30, 200, 200]}
        }
    
    def analyze_soil_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        soil_composition = self.analyze_soil_composition(hsv)
        moisture_level = self.estimate_moisture(hsv)
        nutrient_levels = self.estimate_nutrients(hsv)
        
        soil_health = self.calculate_soil_health(soil_composition, moisture_level, nutrient_levels)
        
        return {
            'composition': soil_composition,
            'moisture_level': moisture_level,
            'nutrients': nutrient_levels,
            'soil_health': soil_health,
            'recommendations': self.generate_soil_recommendations(soil_composition, moisture_level, nutrient_levels)
        }
    
    def analyze_soil_composition(self, hsv_image):
        composition = {}
        
        for soil_type, ranges in self.color_ranges.items():
            lower = np.array(ranges['lower'])
            upper = np.array(ranges['upper'])
            mask = cv2.inRange(hsv_image, lower, upper)
            percentage = np.sum(mask > 0) / (hsv_image.shape[0] * hsv_image.shape[1])
            composition[soil_type] = percentage
        
        total = sum(composition.values())
        if total > 0:
            for key in composition:
                composition[key] = composition[key] / total
        
        return composition
    
    def estimate_moisture(self, hsv_image):
        saturation = hsv_image[:, :, 1].mean()
        value = hsv_image[:, :, 2].mean()
        
        moisture_index = (saturation / 255.0) * (value / 255.0)
        
        if moisture_index < 0.3:
            return 'dry'
        elif moisture_index < 0.6:
            return 'moderate'
        else:
            return 'wet'
    
    def estimate_nutrients(self, hsv_image):
        dark_pixels = np.sum(hsv_image[:, :, 2] < 50)
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        organic_matter = dark_pixels / total_pixels
        
        nitrogen = organic_matter * 100
        phosphorus = organic_matter * 50
        potassium = organic_matter * 75
        
        return {
            'nitrogen': min(nitrogen, 100),
            'phosphorus': min(phosphorus, 100),
            'potassium': min(potassium, 100),
            'organic_matter': organic_matter * 100
        }
    
    def calculate_soil_health(self, composition, moisture, nutrients):
        health_score = 0
        
        loamy_percentage = composition.get('loamy', 0)
        health_score += loamy_percentage * 40
        
        if moisture == 'moderate':
            health_score += 30
        elif moisture == 'wet':
            health_score += 20
        else:
            health_score += 10
        
        avg_nutrients = (nutrients['nitrogen'] + nutrients['phosphorus'] + nutrients['potassium']) / 3
        health_score += (avg_nutrients / 100) * 30
        
        return min(health_score, 100)
    
    def generate_soil_recommendations(self, composition, moisture, nutrients):
        recommendations = []
        
        if composition.get('clay', 0) > 0.6:
            recommendations.append("Add organic matter to improve clay soil drainage")
        
        if composition.get('sandy', 0) > 0.7:
            recommendations.append("Add compost to improve water retention in sandy soil")
        
        if moisture == 'dry':
            recommendations.append("Increase irrigation frequency")
        elif moisture == 'wet':
            recommendations.append("Reduce irrigation and improve drainage")
        
        if nutrients['nitrogen'] < 30:
            recommendations.append("Apply nitrogen-rich fertilizer")
        
        if nutrients['phosphorus'] < 25:
            recommendations.append("Add phosphorus fertilizer")
        
        if nutrients['potassium'] < 35:
            recommendations.append("Apply potassium fertilizer")
        
        return recommendations