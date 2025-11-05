import math
import numpy as np

class GeoUtils:
    def __init__(self):
        self.earth_radius = 6371
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return self.earth_radius * c
    
    def calculate_area(self, coordinates):
        if len(coordinates) < 3:
            return 0
        
        area = 0
        for i in range(len(coordinates)):
            j = (i + 1) % len(coordinates)
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            
            area += math.radians(lon2 - lon1) * (2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
        
        area = abs(area * self.earth_radius ** 2 / 2)
        return area
    
    def convert_to_hectares(self, area_sq_km):
        return area_sq_km * 100