import numpy as np
import cv2
from PIL import Image

class MultispectralProcessor:
    def __init__(self):
        self.band_combinations = {
            'natural_color': [3, 2, 1],
            'false_color': [4, 3, 2],
            'vegetation_analysis': [5, 4, 3],
            'land_water': [5, 6, 4]
        }
    
    def calculate_ndvi(self, red_band, nir_band):
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        denominator = nir + red
        denominator[denominator == 0] = 1e-8
        
        ndvi = (nir - red) / denominator
        
        return np.clip(ndvi, -1, 1)
    
    def calculate_ndwi(self, green_band, nir_band):
        green = green_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        denominator = green + nir
        denominator[denominator == 0] = 1e-8
        
        ndwi = (green - nir) / denominator
        
        return np.clip(ndwi, -1, 1)
    
    def calculate_evi(self, blue_band, red_band, nir_band):
        blue = blue_band.astype(np.float32)
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        numerator = nir - red
        denominator = nir + 6 * red - 7.5 * blue + 1
        
        denominator[denominator == 0] = 1e-8
        
        evi = 2.5 * numerator / denominator
        
        return np.clip(evi, -1, 1)
    
    def process_multispectral_data(self, bands):
        indices = {}
        
        if 'red' in bands and 'nir' in bands:
            indices['ndvi'] = self.calculate_ndvi(bands['red'], bands['nir'])
        
        if 'green' in bands and 'nir' in bands:
            indices['ndwi'] = self.calculate_ndwi(bands['green'], bands['nir'])
        
        if 'blue' in bands and 'red' in bands and 'nir' in bands:
            indices['evi'] = self.calculate_evi(bands['blue'], bands['red'], bands['nir'])
        
        vegetation_health = self.assess_vegetation_health(indices)
        stress_levels = self.detect_stress(bands, indices)
        
        return {
            'vegetation_indices': indices,
            'vegetation_health': vegetation_health,
            'stress_levels': stress_levels,
            'health_assessment': self.interpret_health_scores(vegetation_health)
        }
    
    def assess_vegetation_health(self, indices):
        health_scores = {}
        
        if 'ndvi' in indices:
            ndvi_mean = np.mean(indices['ndvi'])
            if ndvi_mean > 0.6:
                health_scores['ndvi'] = 'excellent'
            elif ndvi_mean > 0.4:
                health_scores['ndvi'] = 'good'
            elif ndvi_mean > 0.2:
                health_scores['ndvi'] = 'moderate'
            else:
                health_scores['ndvi'] = 'poor'
        
        if 'evi' in indices:
            evi_mean = np.mean(indices['evi'])
            if evi_mean > 0.5:
                health_scores['evi'] = 'excellent'
            elif evi_mean > 0.3:
                health_scores['evi'] = 'good'
            elif evi_mean > 0.1:
                health_scores['evi'] = 'moderate'
            else:
                health_scores['evi'] = 'poor'
        
        return health_scores
    
    def detect_stress(self, bands, indices):
        stress_indicators = {}
        
        if 'ndvi' in indices:
            ndvi_std = np.std(indices['ndvi'])
            if ndvi_std > 0.2:
                stress_indicators['water_stress'] = 'high'
            elif ndvi_std > 0.1:
                stress_indicators['water_stress'] = 'moderate'
            else:
                stress_indicators['water_stress'] = 'low'
        
        if 'thermal' in bands:
            thermal_mean = np.mean(bands['thermal'])
            if thermal_mean > 35:
                stress_indicators['heat_stress'] = 'high'
            elif thermal_mean > 30:
                stress_indicators['heat_stress'] = 'moderate'
            else:
                stress_indicators['heat_stress'] = 'low'
        
        return stress_indicators
    
    def interpret_health_scores(self, health_scores):
        interpretations = []
        
        if health_scores.get('ndvi') in ['poor', 'moderate']:
            interpretations.append("Vegetation shows signs of stress - investigate water and nutrient levels")
        
        if health_scores.get('evi') in ['poor', 'moderate']:
            interpretations.append("Canopy health may be compromised - check for disease or pest infestation")
        
        if all(score in ['excellent', 'good'] for score in health_scores.values()):
            interpretations.append("Crop health is excellent - maintain current practices")
        
        return interpretations