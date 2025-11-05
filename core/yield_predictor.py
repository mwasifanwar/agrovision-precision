import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

class YieldLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=1):
        super(YieldLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class YieldPredictor:
    def __init__(self):
        self.lstm_model = YieldLSTM()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, satellite_data, weather_data, soil_data):
        features = []
        
        ndvi_mean = np.mean(satellite_data.get('ndvi', []))
        ndvi_std = np.std(satellite_data.get('ndvi', []))
        
        features.extend([
            ndvi_mean,
            ndvi_std,
            satellite_data.get('vegetation_health', 0),
            satellite_data.get('canopy_cover', 0)
        ])
        
        features.extend([
            weather_data.get('temperature_avg', 0),
            weather_data.get('rainfall_total', 0),
            weather_data.get('sunlight_hours', 0),
            weather_data.get('humidity_avg', 0)
        ])
        
        features.extend([
            soil_data.get('nitrogen', 0),
            soil_data.get('phosphorus', 0),
            soil_data.get('potassium', 0),
            soil_data.get('moisture_level_numeric', 0)
        ])
        
        return np.array(features)
    
    def predict_yield(self, satellite_data, weather_data, soil_data, historical_yields=None):
        features = self.extract_features(satellite_data, weather_data, soil_data)
        
        if not self.is_trained and historical_yields is not None:
            self.train_models(historical_yields, [features])
        
        if self.is_trained:
            rf_prediction = self.rf_model.predict([features])[0]
            xgb_prediction = self.xgb_model.predict([features])[0]
            
            ensemble_prediction = (rf_prediction + xgb_prediction) / 2
            
            confidence = self.calculate_confidence(features)
            
            return {
                'predicted_yield': max(0, ensemble_prediction),
                'confidence': confidence,
                'yield_category': self.categorize_yield(ensemble_prediction),
                'factors': self.analyze_yield_factors(features),
                'recommendations': self.generate_yield_recommendations(features)
            }
        else:
            baseline_prediction = self.baseline_prediction(features)
            return {
                'predicted_yield': baseline_prediction,
                'confidence': 0.5,
                'yield_category': self.categorize_yield(baseline_prediction),
                'factors': self.analyze_yield_factors(features),
                'recommendations': ['Collect more historical data for accurate predictions']
            }
    
    def train_models(self, historical_yields, features_list):
        if len(historical_yields) < 10:
            return
        
        X = np.array(features_list)
        y = np.array(historical_yields)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.rf_model.fit(X_scaled, y)
        self.xgb_model.fit(X_scaled, y)
        
        self.is_trained = True
    
    def calculate_confidence(self, features):
        if not self.is_trained:
            return 0.5
        
        features_scaled = self.scaler.transform([features])
        
        rf_pred = self.rf_model.predict(features_scaled)
        xgb_pred = self.xgb_model.predict(features_scaled)
        
        agreement = 1 - (abs(rf_pred - xgb_pred) / (rf_pred + 1e-8))
        
        return min(agreement[0], 1.0)
    
    def categorize_yield(self, yield_value):
        if yield_value < 2:
            return 'very_low'
        elif yield_value < 4:
            return 'low'
        elif yield_value < 6:
            return 'medium'
        elif yield_value < 8:
            return 'high'
        else:
            return 'very_high'
    
    def analyze_yield_factors(self, features):
        factors = []
        
        if features[0] < 0.3:
            factors.append('Low vegetation index')
        elif features[0] > 0.7:
            factors.append('Excellent vegetation health')
        
        if features[4] < 15:
            factors.append('Low temperature may affect growth')
        elif features[4] > 30:
            factors.append('High temperature stress')
        
        if features[5] < 500:
            factors.append('Insufficient rainfall')
        elif features[5] > 1500:
            factors.append('Excessive rainfall')
        
        if features[8] < 30:
            factors.append('Low nitrogen levels')
        
        return factors
    
    def generate_yield_recommendations(self, features):
        recommendations = []
        
        if features[0] < 0.3:
            recommendations.append("Improve crop health through better nutrient management")
        
        if features[4] < 15:
            recommendations.append("Consider using protective covers during cold periods")
        
        if features[5] < 500:
            recommendations.append("Implement irrigation system to supplement rainfall")
        
        if features[8] < 30:
            recommendations.append("Apply nitrogen fertilizer to boost growth")
        
        if len(recommendations) == 0:
            recommendations.append("Continue current agricultural practices")
        
        return recommendations
    
    def baseline_prediction(self, features):
        ndvi = features[0]
        temperature = features[4]
        rainfall = features[5]
        nitrogen = features[8]
        
        base_yield = 5.0
        yield_adjustment = (ndvi - 0.5) * 2 + (temperature - 20) * 0.1 + (rainfall - 1000) * 0.001 + (nitrogen - 50) * 0.02
        
        return max(0, base_yield + yield_adjustment)