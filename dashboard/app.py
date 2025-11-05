from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import json
import threading
import time

app = Flask(__name__)

class Dashboard:
    def __init__(self):
        self.field_data = []
        self.analysis_results = []
        self.realtime_updates = []
    
    def generate_field_overview(self):
        from agrovision_precision.core.disease_detector import DiseaseDetector
        from agrovision_precision.core.soil_analyzer import SoilAnalyzer
        
        detector = DiseaseDetector()
        soil_analyzer = SoilAnalyzer()
        
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        disease_result = detector.detect_diseases(sample_image)
        soil_result = soil_analyzer.analyze_soil_image(sample_image)
        
        return {
            'disease_analysis': disease_result,
            'soil_analysis': soil_result,
            'field_health': self.calculate_field_health(disease_result, soil_result),
            'timestamp': time.time()
        }
    
    def calculate_field_health(self, disease_result, soil_result):
        health_score = 100
        
        if disease_result['disease'] != 'healthy':
            health_score -= 30
        
        if disease_result['severity'] == 'high':
            health_score -= 20
        elif disease_result['severity'] == 'medium':
            health_score -= 10
        
        if soil_result['soil_health'] < 70:
            health_score -= (70 - soil_result['soil_health'])
        
        return max(health_score, 0)

dashboard = Dashboard()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/field_data')
def get_field_data():
    field_overview = dashboard.generate_field_overview()
    return jsonify(field_overview)

@app.route('/api/analysis_history')
def get_analysis_history():
    return jsonify(dashboard.analysis_results[-50:])

@app.route('/api/realtime_updates')
def get_realtime_updates():
    return jsonify(dashboard.realtime_updates[-20:])

def run_dashboard():
    app.run(host='0.0.0.0', port=5000, debug=True)