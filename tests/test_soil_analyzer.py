import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agrovision_precision.core.soil_analyzer import SoilAnalyzer

class TestSoilAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SoilAnalyzer()
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_soil_analysis(self):
        result = self.analyzer.analyze_soil_image(self.sample_image)
        self.assertIn('composition', result)
        self.assertIn('moisture_level', result)
        self.assertIn('nutrients', result)
    
    def test_soil_health_calculation(self):
        composition = {'clay': 0.3, 'sandy': 0.3, 'loamy': 0.4}
        nutrients = {'nitrogen': 50, 'phosphorus': 40, 'potassium': 45, 'organic_matter': 3.0}
        health = self.analyzer.calculate_soil_health(composition, 'moderate', nutrients)
        self.assertGreaterEqual(health, 0)
        self.assertLessEqual(health, 100)

if __name__ == '__main__':
    unittest.main()