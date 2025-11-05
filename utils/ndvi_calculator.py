import numpy as np
import cv2

class NDVICalculator:
    def __init__(self):
        self.ndvi_ranges = {
            'water': (-1.0, 0.0),
            'bare_soil': (0.0, 0.2),
            'sparse_vegetation': (0.2, 0.5),
            'dense_vegetation': (0.5, 1.0)
        }
    
    def calculate_ndvi(self, red_band, nir_band):
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        denominator = nir + red
        denominator[denominator == 0] = 1e-8
        
        ndvi = (nir - red) / denominator
        return np.clip(ndvi, -1, 1)
    
    def classify_vegetation(self, ndvi_data):
        classification = np.zeros_like(ndvi_data, dtype=np.uint8)
        
        classification[(ndvi_data >= -1.0) & (ndvi_data < 0.0)] = 0
        classification[(ndvi_data >= 0.0) & (ndvi_data < 0.2)] = 1
        classification[(ndvi_data >= 0.2) & (ndvi_data < 0.5)] = 2
        classification[(ndvi_data >= 0.5) & (ndvi_data <= 1.0)] = 3
        
        return classification
    
    def calculate_vegetation_statistics(self, ndvi_data):
        stats = {}
        
        stats['mean_ndvi'] = np.mean(ndvi_data)
        stats['std_ndvi'] = np.std(ndvi_data)
        stats['max_ndvi'] = np.max(ndvi_data)
        stats['min_ndvi'] = np.min(ndvi_data)
        
        classification = self.classify_vegetation(ndvi_data)
        total_pixels = ndvi_data.size
        
        stats['water_coverage'] = np.sum(classification == 0) / total_pixels
        stats['bare_soil_coverage'] = np.sum(classification == 1) / total_pixels
        stats['sparse_vegetation'] = np.sum(classification == 2) / total_pixels
        stats['dense_vegetation'] = np.sum(classification == 3) / total_pixels
        
        return stats