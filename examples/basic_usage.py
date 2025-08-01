#!/usr/bin/env python3
"""
Basic Usage Example for Risk-based Flexible Authentication Framework

This example demonstrates how to use the framework for basic authentication scenarios.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import the framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc_implementation import (
    CustomerBehaviorDataLake, RiskScoringModel, LTMAgent, 
    STMAgent, AuthenticationOrchestrator, CustomerProfile, TransactionContext
)

def basic_authentication_example():
    """Demonstrate basic authentication workflow"""
    print("=" * 60)
    print("BASIC AUTHENTICATION FRAMEWORK USAGE")
    print("=" * 60)
    
    # Step 1: Initialize the framework components
    print("\n1. Initializing Framework Components...")
    data_lake = CustomerBehaviorDataLake()
    risk_model = RiskScoringModel()
    ltm_agent = LTMAgent(data_lake, risk_model)
    stm_agent = STMAgent(ltm_agent)
    orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
    
    # Step 2: Create a sample customer profile
    print("\n2. Creating Sample Customer Profile...")
    customer_id = "CUST_001"
    customer_profile = CustomerProfile(
        customer_id=customer_id,
        behavior_features={
            'total_transactions': 25,
            'avg_transaction_amount': 1500.0,
            'max_transaction_amount': 5000.0,
            'unique_recipients': 8,
            'login_frequency': 15,
            'device_switching_frequency': 2
        },
        digital_features={
            'device_fingerprint': 'abc123def456ghi789',
            'ip_address': '192.168.1.100',
            'geolocation': 'New York, NY'
        },
        risk_score=0.3,
        last_updated=datetime.now() - timedelta(days=1),
        authentication_history=[]
    )
    
    # Add customer to data lake
    data_lake.update_customer_profile(customer_id, customer_profile)
    print(f"   ✓ Customer {customer_id} profile created")
    
    # Step 3: Process a low-risk transaction
    print("\n3. Processing Low-Risk Transaction...")
    low_risk_transaction = TransactionContext(
        transaction_id="TXN_001",
        customer_id=customer_id,
        amount=500.0,
        device_fingerprint='abc123def456ghi789',
        ip_address='192.168.1.100',
        geolocation='New York, NY',
        timestamp=datetime.now(),
        transaction_type='payment',
        recipient_info='REGULAR_MERCHANT'
    )
    
    result_low = orchestrator.process_authentication_request(low_risk_transaction)
    print(f"   Risk Level: {result_low['risk_level'].upper()}")
    print(f"   Authentication Method: {result_low['authentication_method']}")
    print(f"   Processing Time: {result_low['processing_time']:.6f} seconds")
    print(f"   Confidence: {result_low['confidence']:.2f}")
    
    # Step 4: Process a high-risk transaction
    print("\n4. Processing High-Risk Transaction...")
    high_risk_transaction = TransactionContext(
        transaction_id="TXN_002",
        customer_id=customer_id,
        amount=15000.0,
        device_fingerprint='new_device_xyz789',
        ip_address='10.0.0.50',
        geolocation='Unknown_Location',
        timestamp=datetime.now().replace(hour=3),  # 3 AM
        transaction_type='transfer',
        recipient_info='NEW_RECIPIENT'
    )
    
    result_high = orchestrator.process_authentication_request(high_risk_transaction)
    print(f"   Risk Level: {result_high['risk_level'].upper()}")
    print(f"   Authentication Method: {result_high['authentication_method']}")
    print(f"   Processing Time: {result_high['processing_time']:.6f} seconds")
    print(f"   Confidence: {result_high['confidence']:.2f}")
    
    # Step 5: Show framework statistics
    print("\n5. Framework Statistics...")
    customer = data_lake.get_customer_profile(customer_id)
    print(f"   Total Transactions: {customer.behavior_features['total_transactions']}")
    print(f"   Current Risk Score: {customer.risk_score:.3f}")
    print(f"   Authentication History: {len(customer.authentication_history)} entries")
    
    print("\n" + "=" * 60)
    print("BASIC USAGE EXAMPLE COMPLETED")
    print("=" * 60)

def batch_processing_example():
    """Demonstrate batch processing of multiple transactions"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Initialize framework
    data_lake = CustomerBehaviorDataLake()
    risk_model = RiskScoringModel()
    ltm_agent = LTMAgent(data_lake, risk_model)
    stm_agent = STMAgent(ltm_agent)
    orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
    
    # Create multiple customers
    customers = []
    for i in range(5):
        customer_id = f"CUST_{i+1:03d}"
        customer_profile = CustomerProfile(
            customer_id=customer_id,
            behavior_features={
                'total_transactions': 10 + i * 5,
                'avg_transaction_amount': 1000.0 + i * 200,
                'max_transaction_amount': 3000.0 + i * 1000,
                'unique_recipients': 3 + i,
                'login_frequency': 8 + i * 2,
                'device_switching_frequency': i
            },
            digital_features={
                'device_fingerprint': f'device_{i:03d}_fingerprint',
                'ip_address': f'192.168.1.{100 + i}',
                'geolocation': f'City_{i+1}'
            },
            risk_score=0.2 + i * 0.1,
            last_updated=datetime.now() - timedelta(days=i),
            authentication_history=[]
        )
        customers.append(customer_profile)
        data_lake.update_customer_profile(customer_id, customer_profile)
    
    print(f"   ✓ Created {len(customers)} customer profiles")
    
    # Process batch transactions
    transactions = []
    for i, customer in enumerate(customers):
        for j in range(3):  # 3 transactions per customer
            transaction = TransactionContext(
                transaction_id=f"TXN_{i}_{j}",
                customer_id=customer.customer_id,
                amount=500.0 + j * 1000,
                device_fingerprint=customer.digital_features['device_fingerprint'],
                ip_address=customer.digital_features['ip_address'],
                geolocation=customer.digital_features['geolocation'],
                timestamp=datetime.now() - timedelta(hours=j),
                transaction_type='payment' if j % 2 == 0 else 'transfer',
                recipient_info=f'RECIPIENT_{j}'
            )
            transactions.append(transaction)
    
    print(f"   ✓ Created {len(transactions)} transactions")
    
    # Process all transactions
    results = []
    for transaction in transactions:
        result = orchestrator.process_authentication_request(transaction)
        results.append(result)
    
    # Analyze results
    risk_levels = [r['risk_level'] for r in results]
    processing_times = [r['processing_time'] for r in results]
    
    print(f"\n   Batch Processing Results:")
    print(f"   Total Transactions: {len(results)}")
    print(f"   Low Risk: {risk_levels.count('low')}")
    print(f"   Medium Risk: {risk_levels.count('medium')}")
    print(f"   High Risk: {risk_levels.count('high')}")
    print(f"   Average Processing Time: {sum(processing_times)/len(processing_times):.6f} seconds")
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Run basic usage example
    basic_authentication_example()
    
    # Run batch processing example
    batch_processing_example() 