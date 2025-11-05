import torch
import cv2
import argparse
import json
from datetime import datetime

from agrovision_precision.core.disease_detector import DiseaseDetector
from agrovision_precision.core.soil_analyzer import SoilAnalyzer
from agrovision_precision.core.yield_predictor import YieldPredictor
from agrovision_precision.core.irrigation_optimizer import IrrigationOptimizer

def main():
    parser = argparse.ArgumentParser(description='AgroVision Precision Inference')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--analysis_type', type=str, choices=['disease', 'soil', 'yield', 'irrigation'], required=True)
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    print("Loading AgroVision Precision models...")
    
    if args.analysis_type == 'disease':
        analyzer = DiseaseDetector('models/disease_detector.pth')
        result = analyzer.detect_diseases(args.image)
        print("Disease Analysis Results:")
        print(f"Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Severity: {result['severity']}")
        print(f"Recommendation: {result['recommendation']}")
    
    elif args.analysis_type == 'soil':
        analyzer = SoilAnalyzer()
        result = analyzer.analyze_soil_image(args.image)
        print("Soil Analysis Results:")
        print(f"Composition: {result['composition']}")
        print(f"Moisture Level: {result['moisture_level']}")
        print(f"Nutrients: {result['nutrients']}")
        print(f"Soil Health: {result['soil_health']:.1f}%")
        print("Recommendations:", result['recommendations'])
    
    elif args.analysis_type == 'yield':
        predictor = YieldPredictor()
        
        satellite_data = {
            'ndvi': [0.6, 0.65, 0.7],
            'vegetation_health': 0.8,
            'canopy_cover': 0.75
        }
        
        weather_data = {
            'temperature_avg': 25.0,
            'rainfall_total': 800.0,
            'sunlight_hours': 12.0,
            'humidity_avg': 65.0
        }
        
        soil_data = {
            'nitrogen': 45.0,
            'phosphorus': 35.0,
            'potassium': 40.0,
            'moisture_level_numeric': 0.6
        }
        
        result = predictor.predict_yield(satellite_data, weather_data, soil_data)
        print("Yield Prediction Results:")
        print(f"Predicted Yield: {result['predicted_yield']:.2f} tons/hectare")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Yield Category: {result['yield_category']}")
        print("Factors:", result['factors'])
        print("Recommendations:", result['recommendations'])
    
    elif args.analysis_type == 'irrigation':
        optimizer = IrrigationOptimizer()
        
        weather_data = {
            'temperature_avg': 28.0,
            'humidity_avg': 60.0,
            'wind_speed': 3.0,
            'solar_radiation': 18.0
        }
        
        soil_data = {
            'composition': {'clay': 0.3, 'sandy': 0.2, 'loamy': 0.5},
            'moisture_level_numeric': 0.5
        }
        
        result = optimizer.calculate_water_requirements('corn', weather_data, soil_data)
        schedule = optimizer.optimize_irrigation_schedule(result, soil_data, {})
        
        print("Irrigation Optimization Results:")
        print(f"Daily Water Requirement: {result['daily_water_requirement']} mm")
        print(f"Weekly Requirement: {result['weekly_requirement']} mm")
        print("Efficiency Recommendations:", result['efficiency_recommendations'])
        print("Irrigation Schedule:", schedule)
    
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': args.analysis_type,
            'input_image': args.image,
            'results': result
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()