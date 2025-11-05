import yaml
import os

class ConfigLoader:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'disease_detection': {
                'confidence_threshold': 0.7,
                'severity_thresholds': {'low': 0.1, 'medium': 0.3, 'high': 0.5}
            },
            'soil_analysis': {
                'moisture_thresholds': {'dry': 0.3, 'moderate': 0.6},
                'nutrient_thresholds': {'low': 30, 'medium': 50}
            },
            'yield_prediction': {
                'confidence_threshold': 0.6,
                'historical_data_points': 50
            },
            'irrigation': {
                'crop_coefficients': {
                    'wheat': 0.8, 'corn': 1.0, 'soybean': 0.85,
                    'rice': 1.2, 'cotton': 0.9
                }
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default