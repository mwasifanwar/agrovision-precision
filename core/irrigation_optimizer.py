import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class IrrigationOptimizer:
    def __init__(self):
        self.moisture_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.crop_coefficients = {
            'wheat': 0.8,
            'corn': 1.0,
            'soybean': 0.85,
            'rice': 1.2,
            'cotton': 0.9
        }
    
    def calculate_water_requirements(self, crop_type, weather_data, soil_data, crop_stage='mid'):
        crop_coeff = self.crop_coefficients.get(crop_type, 1.0)
        
        stage_coefficients = {
            'initial': 0.5,
            'development': 0.7,
            'mid': 1.0,
            'late': 0.8
        }
        stage_coeff = stage_coefficients.get(crop_stage, 1.0)
        
        temperature = weather_data.get('temperature_avg', 20)
        humidity = weather_data.get('humidity_avg', 60)
        wind_speed = weather_data.get('wind_speed', 2)
        solar_radiation = weather_data.get('solar_radiation', 15)
        
        reference_evapotranspiration = self.calculate_eto(temperature, humidity, wind_speed, solar_radiation)
        
        crop_evapotranspiration = reference_evapotranspiration * crop_coeff * stage_coeff
        
        soil_moisture = soil_data.get('moisture_level_numeric', 0.5)
        soil_type_factor = self.get_soil_type_factor(soil_data.get('composition', {}))
        
        irrigation_need = max(0, crop_evapotranspiration - (soil_moisture * 10))
        irrigation_need *= soil_type_factor
        
        return {
            'daily_water_requirement': round(irrigation_need, 2),
            'weekly_requirement': round(irrigation_need * 7, 2),
            'crop_et': round(crop_evapotranspiration, 2),
            'reference_et': round(reference_evapotranspiration, 2),
            'efficiency_recommendations': self.generate_efficiency_recommendations(irrigation_need, soil_data)
        }
    
    def calculate_eto(self, temperature, humidity, wind_speed, solar_radiation):
        delta = 4098 * (0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))) / ((temperature + 237.3) ** 2)
        
        psychrometric_constant = 0.665 * 0.001 * 101.3
        
        net_radiation = solar_radiation * 0.77 - 40
        
        wind_function = 0.34 * max(wind_speed, 0.5)
        
        numerator = (0.408 * delta * net_radiation + 
                    psychrometric_constant * (900 / (temperature + 273)) * wind_function * (1 - humidity/100))
        denominator = delta + psychrometric_constant * (1 + 0.34 * wind_speed)
        
        eto = numerator / denominator
        
        return max(eto, 0.1)
    
    def get_soil_type_factor(self, composition):
        clay_percentage = composition.get('clay', 0)
        sandy_percentage = composition.get('sandy', 0)
        
        if clay_percentage > 0.6:
            return 0.8
        elif sandy_percentage > 0.6:
            return 1.2
        else:
            return 1.0
    
    def generate_efficiency_recommendations(self, irrigation_need, soil_data):
        recommendations = []
        
        if irrigation_need > 10:
            recommendations.append("Consider drip irrigation for water efficiency")
        
        if soil_data.get('composition', {}).get('sandy', 0) > 0.6:
            recommendations.append("Sandy soil detected - use frequent, light irrigation")
        
        if soil_data.get('composition', {}).get('clay', 0) > 0.6:
            recommendations.append("Clay soil detected - use less frequent, deeper irrigation")
        
        if irrigation_need < 5:
            recommendations.append("Low water requirement - monitor soil moisture closely")
        
        recommendations.append(f"Target soil moisture: {self.calculate_target_moisture(soil_data)}%")
        
        return recommendations
    
    def calculate_target_moisture(self, soil_data):
        base_moisture = 60
        
        composition = soil_data.get('composition', {})
        if composition.get('sandy', 0) > 0.6:
            base_moisture = 70
        elif composition.get('clay', 0) > 0.6:
            base_moisture = 50
        
        return base_moisture
    
    def optimize_irrigation_schedule(self, water_requirement, soil_data, weather_forecast):
        schedule = []
        
        daily_water = water_requirement['daily_water_requirement']
        
        soil_type = 'loamy'
        if soil_data.get('composition', {}).get('sandy', 0) > 0.6:
            soil_type = 'sandy'
        elif soil_data.get('composition', {}).get('clay', 0) > 0.6:
            soil_type = 'clay'
        
        if soil_type == 'sandy':
            frequency = 'daily'
            duration = f"{max(15, daily_water * 2)} minutes"
        elif soil_type == 'clay':
            frequency = 'every_3_days'
            duration = f"{max(30, daily_water * 3 * 1.5)} minutes"
        else:
            frequency = 'every_2_days'
            duration = f"{max(25, daily_water * 2 * 1.2)} minutes"
        
        for day in range(7):
            day_schedule = {
                'day': day + 1,
                'irrigate': False,
                'duration': '0 minutes',
                'water_amount': 0
            }
            
            if frequency == 'daily':
                day_schedule['irrigate'] = True
            elif frequency == 'every_2_days' and day % 2 == 0:
                day_schedule['irrigate'] = True
            elif frequency == 'every_3_days' and day % 3 == 0:
                day_schedule['irrigate'] = True
            
            if day_schedule['irrigate']:
                day_schedule['duration'] = duration
                if frequency == 'daily':
                    day_schedule['water_amount'] = round(daily_water, 2)
                elif frequency == 'every_2_days':
                    day_schedule['water_amount'] = round(daily_water * 2, 2)
                elif frequency == 'every_3_days':
                    day_schedule['water_amount'] = round(daily_water * 3, 2)
            
            schedule.append(day_schedule)
        
        return schedule