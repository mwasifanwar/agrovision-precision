import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agrovision_precision.core.disease_detector import DiseaseDetector

class TestDiseaseDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DiseaseDetector()
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_disease_detection(self):
        result = self.detector.detect_diseases(self.sample_image)
        self.assertIn('disease', result)
        self.assertIn('confidence', result)
        self.assertIn('severity', result)
    
    def test_severity_estimation(self):
        severity = self.detector.estimate_severity(self.sample_image, 'early_blight')
        self.assertIn(severity, ['low', 'medium', 'high', 'none'])

if __name__ == '__main__':
    unittest.main()