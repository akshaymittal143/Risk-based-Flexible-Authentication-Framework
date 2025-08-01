#!/usr/bin/env python3
"""
Advanced Configuration Example for Risk-based Flexible Authentication Framework

This example demonstrates how to customize framework parameters and configurations
for different deployment scenarios.
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add parent directory to path to import the framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc_implementation import (
    CustomerBehaviorDataLake, RiskScoringModel, LTMAgent, 
    STMAgent, AuthenticationOrchestrator, CustomerProfile, TransactionContext
)

class CustomRiskScoringModel(RiskScoringModel):
    """Custom risk scoring model with configurable thresholds"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or self._default_config()
    
    def _default_config(self):
        """Default configuration for risk scoring"""
        return {
            'amount_thresholds': {
                'low': 1000,
                'medium': 5000,
                'high': 15000
            },
            'time_thresholds': {
                'normal_hours_start': 6,
                'normal_hours_end': 23
            },
            'device_fingerprint_min_length': 32,
            'risk_weights': {
                'amount': 0.4,
                'device': 0.3,
                'time': 0.2,
                'location': 0.1
            }
        }
    
    def _rule_based_risk_scoring(self, customer_profile, transaction_context):
        """Custom rule-based risk scoring with configurable parameters"""
        risk_score = 0.0
        
        # Amount-based risk with configurable thresholds
        amount = transaction_context.amount
        if amount > self.config['amount_thresholds']['high']:
            risk_score += self.config['risk_weights']['amount']
        elif amount > self.config['amount_thresholds']['medium']:
            risk_score += self.config['risk_weights']['amount'] * 0.5
            
        # Device fingerprint risk
        if len(transaction_context.device_fingerprint) < self.config['device_fingerprint_min_length']:
            risk_score += self.config['risk_weights']['device']
            
        # Time-based risk
        hour = transaction_context.timestamp.hour
        if hour < self.config['time_thresholds']['normal_hours_start'] or \
           hour > self.config['time_thresholds']['normal_hours_end']:
            risk_score += self.config['risk_weights']['time']
            
        # Location-based risk
        if 'Unknown' in transaction_context.geolocation:
            risk_score += self.config['risk_weights']['location']
            
        # Determine risk level
        if risk_score >= 0.7:
            return 'high', risk_score
        elif risk_score >= 0.4:
            return 'medium', risk_score
        else:
            return 'low', risk_score

class CustomLTMAgent(LTMAgent):
    """Custom LTM agent with enhanced behavioral analysis"""
    
    def __init__(self, data_lake, risk_model, config=None):
        super().__init__(data_lake, risk_model)
        self.config = config or self._default_config()
    
    def _default_config(self):
        """Default configuration for LTM agent"""
        return {
            'anomaly_thresholds': {
                'unusual_amount_multiplier': 3.0,
                'rapid_device_switching': 5,
                'failed_auth_threshold': 3
            },
            'behavioral_weights': {
                'transaction_frequency': 0.3,
                'amount_patterns': 0.4,
                'device_usage': 0.3
            }
        }
    
    def _detect_anomalies(self, customer_id):
        """Enhanced anomaly detection with configurable thresholds"""
        profile = self.data_lake.get_customer_profile(customer_id)
        anomalies = []
        
        if profile:
            # Check for unusual transaction amounts
            avg_amount = profile.behavior_features.get('avg_transaction_amount', 0)
            max_amount = profile.behavior_features.get('max_transaction_amount', 0)
            
            if max_amount > avg_amount * self.config['anomaly_thresholds']['unusual_amount_multiplier']:
                anomalies.append('unusually_high_transaction_amount')
                
            # Check for rapid device switching
            device_switching = profile.behavior_features.get('device_switching_frequency', 0)
            if device_switching > self.config['anomaly_thresholds']['rapid_device_switching']:
                anomalies.append('rapid_device_switching')
                
            # Check for multiple failed authentications
            failed_auths = [auth for auth in profile.authentication_history 
                           if auth.get('status') == 'failed']
            if len(failed_auths) > self.config['anomaly_thresholds']['failed_auth_threshold']:
                anomalies.append('multiple_failed_authentications')
                
        return anomalies

def financial_institution_config():
    """Configuration optimized for financial institutions"""
    return {
        'risk_scoring': {
            'amount_thresholds': {
                'low': 500,
                'medium': 2500,
                'high': 10000
            },
            'time_thresholds': {
                'normal_hours_start': 7,
                'normal_hours_end': 22
            },
            'device_fingerprint_min_length': 64,
            'risk_weights': {
                'amount': 0.5,
                'device': 0.2,
                'time': 0.2,
                'location': 0.1
            }
        },
        'ltm_agent': {
            'anomaly_thresholds': {
                'unusual_amount_multiplier': 2.5,
                'rapid_device_switching': 3,
                'failed_auth_threshold': 2
            },
            'behavioral_weights': {
                'transaction_frequency': 0.4,
                'amount_patterns': 0.4,
                'device_usage': 0.2
            }
        }
    }

def ecommerce_config():
    """Configuration optimized for e-commerce platforms"""
    return {
        'risk_scoring': {
            'amount_thresholds': {
                'low': 2000,
                'medium': 8000,
                'high': 20000
            },
            'time_thresholds': {
                'normal_hours_start': 5,
                'normal_hours_end': 24
            },
            'device_fingerprint_min_length': 32,
            'risk_weights': {
                'amount': 0.3,
                'device': 0.3,
                'time': 0.2,
                'location': 0.2
            }
        },
        'ltm_agent': {
            'anomaly_thresholds': {
                'unusual_amount_multiplier': 4.0,
                'rapid_device_switching': 8,
                'failed_auth_threshold': 5
            },
            'behavioral_weights': {
                'transaction_frequency': 0.2,
                'amount_patterns': 0.5,
                'device_usage': 0.3
            }
        }
    }

def advanced_configuration_example():
    """Demonstrate advanced configuration capabilities"""
    print("=" * 70)
    print("ADVANCED CONFIGURATION EXAMPLE")
    print("=" * 70)
    
    # Test different configurations
    configurations = {
        'Financial Institution': financial_institution_config(),
        'E-commerce Platform': ecommerce_config()
    }
    
    for config_name, config in configurations.items():
        print(f"\n📋 Testing {config_name} Configuration")
        print("-" * 50)
        
        # Initialize framework with custom configuration
        data_lake = CustomerBehaviorDataLake()
        
        # Custom risk model
        risk_model = CustomRiskScoringModel(config['risk_scoring'])
        
        # Custom LTM agent
        ltm_agent = CustomLTMAgent(data_lake, risk_model, config['ltm_agent'])
        
        # Standard STM agent and orchestrator
        stm_agent = STMAgent(ltm_agent)
        orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
        
        # Create test customer
        customer_id = f"TEST_{config_name.replace(' ', '_')}"
        customer_profile = CustomerProfile(
            customer_id=customer_id,
            behavior_features={
                'total_transactions': 50,
                'avg_transaction_amount': 3000.0,
                'max_transaction_amount': 12000.0,
                'unique_recipients': 15,
                'login_frequency': 25,
                'device_switching_frequency': 4
            },
            digital_features={
                'device_fingerprint': 'test_device_fingerprint_64_chars_long_for_testing_purposes',
                'ip_address': '192.168.1.200',
                'geolocation': 'San Francisco, CA'
            },
            risk_score=0.4,
            last_updated=datetime.now() - timedelta(days=2),
            authentication_history=[]
        )
        data_lake.update_customer_profile(customer_id, customer_profile)
        
        # Test transactions
        test_transactions = [
            {
                'name': 'Normal Transaction',
                'amount': 1000,
                'device': 'test_device_fingerprint_64_chars_long_for_testing_purposes',
                'time': datetime.now().replace(hour=14),
                'location': 'San Francisco, CA'
            },
            {
                'name': 'High Amount Transaction',
                'amount': 15000,
                'device': 'test_device_fingerprint_64_chars_long_for_testing_purposes',
                'time': datetime.now().replace(hour=14),
                'location': 'San Francisco, CA'
            },
            {
                'name': 'Late Night Transaction',
                'amount': 5000,
                'device': 'test_device_fingerprint_64_chars_long_for_testing_purposes',
                'time': datetime.now().replace(hour=2),
                'location': 'San Francisco, CA'
            },
            {
                'name': 'Unknown Location Transaction',
                'amount': 3000,
                'device': 'test_device_fingerprint_64_chars_long_for_testing_purposes',
                'time': datetime.now().replace(hour=14),
                'location': 'Unknown_Location'
            }
        ]
        
        print(f"   Configuration Parameters:")
        print(f"   - Amount Thresholds: {config['risk_scoring']['amount_thresholds']}")
        print(f"   - Time Thresholds: {config['risk_scoring']['time_thresholds']}")
        print(f"   - Risk Weights: {config['risk_scoring']['risk_weights']}")
        
        print(f"\n   Test Results:")
        for i, test_txn in enumerate(test_transactions):
            transaction = TransactionContext(
                transaction_id=f"TEST_{i+1}",
                customer_id=customer_id,
                amount=test_txn['amount'],
                device_fingerprint=test_txn['device'],
                ip_address='192.168.1.200',
                geolocation=test_txn['location'],
                timestamp=test_txn['time'],
                transaction_type='payment',
                recipient_info='TEST_MERCHANT'
            )
            
            result = orchestrator.process_authentication_request(transaction)
            print(f"   {test_txn['name']}: {result['risk_level'].upper()} risk "
                  f"({result['authentication_method']})")
    
    print("\n" + "=" * 70)
    print("ADVANCED CONFIGURATION EXAMPLE COMPLETED")
    print("=" * 70)

def configuration_export_example():
    """Demonstrate configuration export and import"""
    print("\n" + "=" * 70)
    print("CONFIGURATION EXPORT/IMPORT EXAMPLE")
    print("=" * 70)
    
    # Create a custom configuration
    custom_config = {
        'name': 'Custom Banking Configuration',
        'version': '1.0',
        'description': 'Optimized for high-security banking applications',
        'risk_scoring': financial_institution_config()['risk_scoring'],
        'ltm_agent': financial_institution_config()['ltm_agent'],
        'metadata': {
            'created_by': 'Security Team',
            'created_date': datetime.now().isoformat(),
            'environment': 'production'
        }
    }
    
    # Export configuration to JSON
    config_file = 'custom_config.json'
    with open(config_file, 'w') as f:
        json.dump(custom_config, f, indent=2, default=str)
    
    print(f"   ✓ Configuration exported to {config_file}")
    
    # Import configuration
    with open(config_file, 'r') as f:
        imported_config = json.load(f)
    
    print(f"   ✓ Configuration imported from {config_file}")
    print(f"   Configuration Name: {imported_config['name']}")
    print(f"   Version: {imported_config['version']}")
    print(f"   Description: {imported_config['description']}")
    
    # Clean up
    os.remove(config_file)
    print(f"   ✓ Cleaned up {config_file}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION EXPORT/IMPORT EXAMPLE COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    # Run advanced configuration example
    advanced_configuration_example()
    
    # Run configuration export/import example
    configuration_export_example() 