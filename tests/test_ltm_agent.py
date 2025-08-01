#!/usr/bin/env python3
"""
Unit tests for LTM Agent component
"""

import sys
import os
import unittest
from datetime import datetime, timedelta

# Add parent directory to path to import the framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc_implementation import (
    CustomerBehaviorDataLake, RiskScoringModel, LTMAgent, CustomerProfile
)

class TestLTMAgent(unittest.TestCase):
    """Test cases for LTM Agent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_lake = CustomerBehaviorDataLake()
        self.risk_model = RiskScoringModel()
        self.ltm_agent = LTMAgent(self.data_lake, self.risk_model)
        
        # Create test customer profile
        self.customer_id = "TEST_CUST_001"
        self.customer_profile = CustomerProfile(
            customer_id=self.customer_id,
            behavior_features={
                'total_transactions': 30,
                'avg_transaction_amount': 2000.0,
                'max_transaction_amount': 8000.0,
                'unique_recipients': 10,
                'login_frequency': 20,
                'device_switching_frequency': 3
            },
            digital_features={
                'device_fingerprint': 'test_device_fingerprint_123',
                'ip_address': '192.168.1.100',
                'geolocation': 'Test City, State'
            },
            risk_score=0.4,
            last_updated=datetime.now() - timedelta(days=1),
            authentication_history=[]
        )
        
        # Add customer to data lake
        self.data_lake.update_customer_profile(self.customer_id, self.customer_profile)
    
    def test_analyze_customer_behavior(self):
        """Test customer behavior analysis"""
        analysis = self.ltm_agent.analyze_customer_behavior(self.customer_id)
        
        # Check that analysis contains expected keys
        self.assertIn('risk_level', analysis)
        self.assertIn('confidence', analysis)
        self.assertIn('anomalies', analysis)
        self.assertIn('behavioral_patterns', analysis)
        
        # Check that risk level is valid
        self.assertIn(analysis['risk_level'], ['low', 'medium', 'high'])
        
        # Check that confidence is between 0 and 1
        self.assertGreaterEqual(analysis['confidence'], 0.0)
        self.assertLessEqual(analysis['confidence'], 1.0)
        
        # Check that anomalies is a list
        self.assertIsInstance(analysis['anomalies'], list)
        
        # Check that behavioral_patterns contains expected features
        self.assertIn('total_transactions', analysis['behavioral_patterns'])
        self.assertIn('avg_transaction_amount', analysis['behavioral_patterns'])
    
    def test_analyze_nonexistent_customer(self):
        """Test behavior analysis for non-existent customer"""
        analysis = self.ltm_agent.analyze_customer_behavior("NONEXISTENT_CUSTOMER")
        
        # Should return high risk for non-existent customers
        self.assertEqual(analysis['risk_level'], 'high')
        self.assertEqual(analysis['confidence'], 0.0)
        self.assertEqual(analysis['anomalies'], [])
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        anomalies = self.ltm_agent._detect_anomalies(self.customer_id)
        
        # Should return a list
        self.assertIsInstance(anomalies, list)
        
        # For normal customer, should have no anomalies initially
        self.assertEqual(len(anomalies), 0)
    
    def test_detect_anomalies_with_high_amount(self):
        """Test anomaly detection with unusually high transaction amount"""
        # Update customer profile with unusually high amount
        self.customer_profile.behavior_features['max_transaction_amount'] = 100000.0
        self.data_lake.update_customer_profile(self.customer_id, self.customer_profile)
        
        anomalies = self.ltm_agent._detect_anomalies(self.customer_id)
        
        # Should detect unusually high transaction amount
        self.assertIn('unusually_high_transaction_amount', anomalies)
    
    def test_detect_anomalies_with_device_switching(self):
        """Test anomaly detection with rapid device switching"""
        # Update customer profile with rapid device switching
        self.customer_profile.behavior_features['device_switching_frequency'] = 10
        self.data_lake.update_customer_profile(self.customer_id, self.customer_profile)
        
        anomalies = self.ltm_agent._detect_anomalies(self.customer_id)
        
        # Should detect rapid device switching
        self.assertIn('rapid_device_switching', anomalies)
    
    def test_detect_anomalies_with_failed_auths(self):
        """Test anomaly detection with multiple failed authentications"""
        # Add failed authentication history
        self.customer_profile.authentication_history = [
            {'status': 'failed', 'timestamp': datetime.now()},
            {'status': 'failed', 'timestamp': datetime.now()},
            {'status': 'failed', 'timestamp': datetime.now()},
            {'status': 'failed', 'timestamp': datetime.now()}
        ]
        self.data_lake.update_customer_profile(self.customer_id, self.customer_profile)
        
        anomalies = self.ltm_agent._detect_anomalies(self.customer_id)
        
        # Should detect multiple failed authentications
        self.assertIn('multiple_failed_authentications', anomalies)
    
    def test_calculate_behavioral_risk(self):
        """Test behavioral risk calculation"""
        risk_level = self.ltm_agent._calculate_behavioral_risk(self.customer_profile)
        
        # Should return valid risk level
        self.assertIn(risk_level, ['low', 'medium', 'high'])
    
    def test_calculate_behavioral_risk_high_amount(self):
        """Test behavioral risk calculation with high transaction amount"""
        # Update customer profile with high amount
        self.customer_profile.behavior_features['max_transaction_amount'] = 15000.0
        self.data_lake.update_customer_profile(self.customer_id, self.customer_profile)
        
        risk_level = self.ltm_agent._calculate_behavioral_risk(self.customer_profile)
        
        # Should be medium or high risk
        self.assertIn(risk_level, ['medium', 'high'])
    
    def test_calculate_behavioral_risk_device_switching(self):
        """Test behavioral risk calculation with device switching"""
        # Update customer profile with device switching
        self.customer_profile.behavior_features['device_switching_frequency'] = 5
        self.data_lake.update_customer_profile(self.customer_id, self.customer_profile)
        
        risk_level = self.ltm_agent._calculate_behavioral_risk(self.customer_profile)
        
        # Should be medium or high risk
        self.assertIn(risk_level, ['medium', 'high'])

if __name__ == '__main__':
    unittest.main() 