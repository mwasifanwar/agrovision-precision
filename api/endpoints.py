from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import cv2
import numpy as np

router = APIRouter()

class AnalysisRequest(BaseModel):
    image_type: str
    coordinates: List[float]
    crop_type: str = None

@router.post("/analyze/disease")
async def analyze_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        from agrovision_precision.core.disease_detector import DiseaseDetector
        detector = DiseaseDetector()
        result = detector.detect_diseases(image)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/soil")
async def analyze_soil(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        from agrovision_precision.core.soil_analyzer import SoilAnalyzer
        analyzer = SoilAnalyzer()
        result = analyzer.analyze_soil_image(image)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/yield")
async def predict_yield(request: AnalysisRequest):
    try:
        from agrovision_precision.core.yield_predictor import YieldPredictor
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
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/irrigation")
async def optimize_irrigation(request: AnalysisRequest):
    try:
        from agrovision_precision.core.irrigation_optimizer import IrrigationOptimizer
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
        
        water_req = optimizer.calculate_water_requirements(
            request.crop_type or 'corn', 
            weather_data, 
            soil_data
        )
        
        schedule = optimizer.optimize_irrigation_schedule(water_req, soil_data, {})
        
        return {
            'water_requirements': water_req,
            'irrigation_schedule': schedule
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api/websocket_handler.py
from fastapi import WebSocket
import json

class WebSocketHandler:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_analysis_update(self, analysis_data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "analysis_update",
                    "data": analysis_data
                })
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.active_connections.remove(connection)