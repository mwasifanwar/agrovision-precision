import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from PIL import Image

class Visualization:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'healthy': 'green',
            'diseased': 'red',
            'soil_dry': 'brown',
            'soil_wet': 'blue'
        }
    
    def plot_disease_detection(self, image, detection_result):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        disease = detection_result['disease']
        confidence = detection_result['confidence']
        severity = detection_result['severity']
        
        colors = {'low': 'yellow', 'medium': 'orange', 'high': 'red'}
        severity_color = colors.get(severity, 'gray')
        
        ax2.bar(['Disease Confidence'], [confidence], color=severity_color, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Confidence Score')
        ax2.set_title(f'Detection: {disease} (Severity: {severity})')
        
        plt.tight_layout()
        return fig
    
    def create_ndvi_heatmap(self, ndvi_data):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(ndvi_data, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title('NDVI Vegetation Index Heatmap')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDVI Value')
        
        healthy_pixels = np.sum(ndvi_data > 0.3)
        total_pixels = ndvi_data.size
        health_percentage = (healthy_pixels / total_pixels) * 100
        
        ax.text(0.02, 0.98, f'Vegetation Health: {health_percentage:.1f}%', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        return fig
    
    def plot_soil_analysis(self, soil_analysis):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        composition = soil_analysis['composition']
        ax1.pie(composition.values(), labels=composition.keys(), autopct='%1.1f%%')
        ax1.set_title('Soil Composition')
        
        nutrients = soil_analysis['nutrients']
        nutrient_names = list(nutrients.keys())[:3]
        nutrient_values = [nutrients[name] for name in nutrient_names]
        ax2.bar(nutrient_names, nutrient_values, color=['blue', 'green', 'orange'])
        ax2.set_ylim(0, 100)
        ax2.set_title('Nutrient Levels (%)')
        
        moisture_level = soil_analysis['moisture_level']
        moisture_colors = {'dry': 'brown', 'moderate': 'yellow', 'wet': 'blue'}
        ax3.bar(['Moisture'], [1], color=moisture_colors.get(moisture_level, 'gray'))
        ax3.set_ylim(0, 1)
        ax3.set_title(f'Moisture Level: {moisture_level}')
        
        soil_health = soil_analysis['soil_health']
        ax4.bar(['Soil Health'], [soil_health], color='green' if soil_health > 70 else 'orange')
        ax4.set_ylim(0, 100)
        ax4.set_title(f'Overall Health: {soil_health:.1f}%')
        
        plt.tight_layout()
        return fig