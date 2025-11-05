import rasterio
import numpy as np
from PIL import Image
import requests
import json

class SatelliteLoader:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.bands = {
            'coastal': 1, 'blue': 2, 'green': 3, 'red': 4,
            'nir': 5, 'swir1': 6, 'swir2': 7, 'panchromatic': 8
        }
    
    def load_sentinel_data(self, file_path):
        with rasterio.open(file_path) as src:
            bands_data = {}
            for band_name, band_number in self.bands.items():
                if band_number <= src.count:
                    bands_data[band_name] = src.read(band_number)
            
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height
            }
        
        return bands_data, metadata
    
    def download_satellite_imagery(self, coordinates, date_range, cloud_cover=10):
        if not self.api_key:
            raise ValueError("API key required for satellite imagery download")
        
        bbox = self._coordinates_to_bbox(coordinates)
        
        search_params = {
            'bbox': bbox,
            'date': f"{date_range[0]}/{date_range[1]}",
            'cloudCoverPercentage': f"[0,{cloud_cover}]",
            'limit': 1
        }
        
        return self._make_api_request(search_params)
    
    def _coordinates_to_bbox(self, coordinates):
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        return f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    def _make_api_request(self, params):
        pass