import cv2
import numpy as np
from PIL import Image
import exifread

class DroneProcessor:
    def __init__(self):
        self.ortho_correction_params = {}
    
    def process_drone_imagery(self, image_path, geotag=True):
        image = cv2.imread(image_path)
        
        if geotag:
            metadata = self.extract_geotags(image_path)
        else:
            metadata = {}
        
        corrected_image = self.ortho_correction(image)
        enhanced_image = self.enhance_agricultural_features(corrected_image)
        
        return {
            'image': enhanced_image,
            'metadata': metadata,
            'processed_features': self.extract_agricultural_features(enhanced_image)
        }
    
    def extract_geotags(self, image_path):
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        
        geotags = {}
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            geotags['latitude'] = self._convert_to_degrees(tags['GPS GPSLatitude'].values)
            geotags['longitude'] = self._convert_to_degrees(tags['GPS GPSLongitude'].values)
        
        if 'GPS GPSAltitude' in tags:
            geotags['altitude'] = float(tags['GPS GPSAltitude'].values[0])
        
        return geotags
    
    def _convert_to_degrees(self, value):
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    
    def ortho_correction(self, image):
        if len(self.ortho_correction_params) == 0:
            self._calibrate_ortho_params(image)
        
        h, w = image.shape[:2]
        camera_matrix = self.ortho_correction_params.get('camera_matrix', np.eye(3))
        dist_coeffs = self.ortho_correction_params.get('dist_coeffs', np.zeros(5))
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        corrected = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        x, y, w, h = roi
        corrected = corrected[y:y+h, x:x+w]
        
        return corrected
    
    def _calibrate_ortho_params(self, image):
        self.ortho_correction_params = {
            'camera_matrix': np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]]),
            'dist_coeffs': np.array([0.1, -0.2, 0, 0, 0.1])
        }
    
    def enhance_agricultural_features(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def extract_agricultural_features(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, (25, 40, 40), (85, 255, 255))
        brown_mask = cv2.inRange(hsv, (10, 50, 50), (20, 255, 255))
        
        total_pixels = image.shape[0] * image.shape[1]
        green_percentage = np.sum(green_mask > 0) / total_pixels
        brown_percentage = np.sum(brown_mask > 0) / total_pixels
        
        return {
            'vegetation_coverage': green_percentage,
            'soil_exposure': brown_percentage,
            'health_indicator': green_percentage / (green_percentage + brown_percentage + 1e-8)
        }