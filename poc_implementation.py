#!/usr/bin/env python3
"""
Proof-of-Concept Implementation for Risk-based Flexible Authentication Framework
Using AI-Agents for Identity and Authentication Systems

This implementation demonstrates the core components described in the paper:
1. Customer Behavior Data Lake
2. Risk-Scoring ML Model
3. Long-Term Memory (LTM) Agent
4. Short-Term Memory (STM) Agent
5. Authentication Orchestration Layer
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CustomerProfile:
    """Customer behavioral and digital profile data structure"""
    customer_id: str
    behavior_features: Dict[str, float]
    digital_features: Dict[str, str]
    risk_score: float
    last_updated: datetime
    authentication_history: List[Dict]

@dataclass
class TransactionContext:
    """Real-time transaction context data"""
    transaction_id: str
    customer_id: str
    amount: float
    device_fingerprint: str
    ip_address: str
    geolocation: str
    timestamp: datetime
    transaction_type: str
    recipient_info: Optional[str] = None

class CustomerBehaviorDataLake:
    """Customer behavior data lake for storing historical behavioral patterns"""
    
    def __init__(self):
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.behavior_events: List[Dict] = []
        self.digital_events: List[Dict] = []
        
    def add_behavior_event(self, customer_id: str, event_type: str, 
                          event_data: Dict, timestamp: datetime):
        """Add behavioral event to data lake"""
        event = {
            'customer_id': customer_id,
            'event_type': event_type,
            'event_data': event_data,
            'timestamp': timestamp
        }
        self.behavior_events.append(event)
        
    def add_digital_event(self, customer_id: str, device_fingerprint: str,
                         digital_data: Dict, timestamp: datetime):
        """Add digital event to data lake"""
        event = {
            'customer_id': customer_id,
            'device_fingerprint': device_fingerprint,
            'digital_data': digital_data,
            'timestamp': timestamp
        }
        self.digital_events.append(event)
        
    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Retrieve customer profile from data lake"""
        return self.customer_profiles.get(customer_id)
        
    def update_customer_profile(self, customer_id: str, profile: CustomerProfile):
        """Update customer profile in data lake"""
        self.customer_profiles[customer_id] = profile
        
    def get_behavioral_features(self, customer_id: str, time_window: timedelta) -> Dict[str, float]:
        """Extract behavioral features for a customer within time window"""
        cutoff_time = datetime.now() - time_window
        customer_events = [e for e in self.behavior_events 
                          if e['customer_id'] == customer_id and e['timestamp'] >= cutoff_time]
        
        features = {
            'total_transactions': len(customer_events),
            'avg_transaction_amount': 0.0,
            'max_transaction_amount': 0.0,
            'unique_recipients': 0,
            'login_frequency': 0,
            'device_switching_frequency': 0
        }
        
        if customer_events:
            amounts = [e['event_data'].get('amount', 0) for e in customer_events 
                      if 'amount' in e['event_data']]
            if amounts:
                features['avg_transaction_amount'] = np.mean(amounts)
                features['max_transaction_amount'] = np.max(amounts)
            
            recipients = set(e['event_data'].get('recipient', '') for e in customer_events 
                           if 'recipient' in e['event_data'])
            features['unique_recipients'] = len(recipients)
            
            login_events = [e for e in customer_events if e['event_type'] == 'login']
            features['login_frequency'] = len(login_events)
            
        return features

class RiskScoringModel:
    """Machine Learning model for risk scoring"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, customer_profile: CustomerProfile, 
                        transaction_context: TransactionContext) -> np.ndarray:
        """Prepare feature vector for risk scoring"""
        features = []
        
        # Behavioral features
        features.extend([
            customer_profile.behavior_features.get('total_transactions', 0),
            customer_profile.behavior_features.get('avg_transaction_amount', 0),
            customer_profile.behavior_features.get('max_transaction_amount', 0),
            customer_profile.behavior_features.get('unique_recipients', 0),
            customer_profile.behavior_features.get('login_frequency', 0)
        ])
        
        # Transaction context features
        features.extend([
            transaction_context.amount,
            len(transaction_context.device_fingerprint),
            len(transaction_context.ip_address),
            (datetime.now() - customer_profile.last_updated).total_seconds()
        ])
        
        # Risk score features
        features.append(customer_profile.risk_score)
        
        return np.array(features).reshape(1, -1)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the risk scoring model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def predict_risk(self, customer_profile: CustomerProfile, 
                    transaction_context: TransactionContext) -> Tuple[str, float]:
        """Predict risk level and confidence score"""
        if not self.is_trained:
            # Fallback to rule-based scoring
            return self._rule_based_risk_scoring(customer_profile, transaction_context)
            
        features = self.prepare_features(customer_profile, transaction_context)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled)[0])
        
        risk_levels = ['low', 'medium', 'high']
        return risk_levels[prediction], confidence
        
    def _rule_based_risk_scoring(self, customer_profile: CustomerProfile,
                                transaction_context: TransactionContext) -> Tuple[str, float]:
        """Rule-based risk scoring as fallback"""
        risk_score = 0.0
        
        # Amount-based risk
        if transaction_context.amount > 10000:
            risk_score += 0.4
        elif transaction_context.amount > 5000:
            risk_score += 0.2
            
        # Device fingerprint risk
        if len(transaction_context.device_fingerprint) < 32:
            risk_score += 0.3
            
        # Behavioral risk
        if customer_profile.behavior_features.get('max_transaction_amount', 0) < transaction_context.amount:
            risk_score += 0.3
            
        # Time-based risk
        time_diff = (datetime.now() - customer_profile.last_updated).total_seconds()
        if time_diff > 86400:  # 24 hours
            risk_score += 0.2
            
        if risk_score >= 0.7:
            return 'high', risk_score
        elif risk_score >= 0.4:
            return 'medium', risk_score
        else:
            return 'low', risk_score

class LTMAgent:
    """Long-Term Memory Agent for historical context and behavioral analysis"""
    
    def __init__(self, data_lake: CustomerBehaviorDataLake, risk_model: RiskScoringModel):
        self.data_lake = data_lake
        self.risk_model = risk_model
        self.customer_personas: Dict[str, Dict] = {}
        
    def analyze_customer_behavior(self, customer_id: str) -> Dict[str, any]:
        """Analyze customer behavior patterns using historical data"""
        profile = self.data_lake.get_customer_profile(customer_id)
        if not profile:
            return {'risk_level': 'high', 'confidence': 0.0, 'anomalies': []}
            
        # Analyze behavioral patterns
        anomalies = self._detect_anomalies(customer_id)
        
        # Calculate behavioral risk score
        behavior_risk = self._calculate_behavioral_risk(profile)
        
        return {
            'risk_level': behavior_risk,
            'confidence': 0.8,  # Placeholder confidence
            'anomalies': anomalies,
            'behavioral_patterns': profile.behavior_features
        }
        
    def _detect_anomalies(self, customer_id: str) -> List[str]:
        """Detect behavioral anomalies"""
        profile = self.data_lake.get_customer_profile(customer_id)
        anomalies = []
        
        if profile:
            # Check for unusual transaction amounts
            if profile.behavior_features.get('max_transaction_amount', 0) > 50000:
                anomalies.append('unusually_high_transaction_amount')
                
            # Check for rapid device switching
            if profile.behavior_features.get('device_switching_frequency', 0) > 5:
                anomalies.append('rapid_device_switching')
                
            # Check for multiple failed authentications
            failed_auths = [auth for auth in profile.authentication_history 
                           if auth.get('status') == 'failed']
            if len(failed_auths) > 3:
                anomalies.append('multiple_failed_authentications')
                
        return anomalies
        
    def _calculate_behavioral_risk(self, profile: CustomerProfile) -> str:
        """Calculate behavioral risk level"""
        risk_factors = 0
        
        if profile.behavior_features.get('max_transaction_amount', 0) > 10000:
            risk_factors += 1
            
        if profile.behavior_features.get('device_switching_frequency', 0) > 3:
            risk_factors += 1
            
        if len(profile.authentication_history) > 10:
            recent_failures = [auth for auth in profile.authentication_history[-5:] 
                              if auth.get('status') == 'failed']
            if len(recent_failures) > 2:
                risk_factors += 1
                
        if risk_factors >= 2:
            return 'high'
        elif risk_factors >= 1:
            return 'medium'
        else:
            return 'low'

class STMAgent:
    """Short-Term Memory Agent for real-time decision making"""
    
    def __init__(self, ltm_agent: LTMAgent):
        self.ltm_agent = ltm_agent
        self.real_time_context: Dict[str, Dict] = {}
        
    def evaluate_transaction(self, transaction_context: TransactionContext) -> Dict[str, any]:
        """Evaluate transaction in real-time context"""
        customer_id = transaction_context.customer_id
        
        # Get LTM analysis
        ltm_analysis = self.ltm_agent.analyze_customer_behavior(customer_id)
        
        # Analyze real-time context
        real_time_risk = self._analyze_real_time_context(transaction_context)
        
        # Combine LTM and STM analysis
        combined_risk = self._combine_risk_assessments(ltm_analysis, real_time_risk)
        
        # Determine authentication requirements
        auth_requirements = self._determine_auth_requirements(combined_risk)
        
        return {
            'risk_level': combined_risk['level'],
            'confidence': combined_risk['confidence'],
            'authentication_requirements': auth_requirements,
            'ltm_analysis': ltm_analysis,
            'real_time_analysis': real_time_risk
        }
        
    def _analyze_real_time_context(self, transaction_context: TransactionContext) -> Dict[str, any]:
        """Analyze real-time transaction context"""
        risk_factors = 0
        anomalies = []
        
        # Amount-based risk
        if transaction_context.amount > 10000:
            risk_factors += 2
            anomalies.append('high_value_transaction')
        elif transaction_context.amount > 5000:
            risk_factors += 1
            
        # Time-based risk (outside normal hours)
        hour = transaction_context.timestamp.hour
        if hour < 6 or hour > 23:
            risk_factors += 1
            anomalies.append('unusual_time')
            
        # Device fingerprint risk
        if len(transaction_context.device_fingerprint) < 32:
            risk_factors += 2
            anomalies.append('weak_device_fingerprint')
            
        # Geolocation risk
        if 'unknown' in transaction_context.geolocation.lower():
            risk_factors += 1
            anomalies.append('unknown_location')
            
        # Determine risk level
        if risk_factors >= 3:
            risk_level = 'high'
        elif risk_factors >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'anomalies': anomalies,
            'confidence': 0.9
        }
        
    def _combine_risk_assessments(self, ltm_analysis: Dict, real_time_analysis: Dict) -> Dict[str, any]:
        """Combine LTM and STM risk assessments"""
        ltm_risk_scores = {'low': 1, 'medium': 2, 'high': 3}
        stm_risk_scores = {'low': 1, 'medium': 2, 'high': 3}
        
        combined_score = (ltm_risk_scores[ltm_analysis['risk_level']] + 
                         stm_risk_scores[real_time_analysis['risk_level']]) / 2
        
        if combined_score >= 2.5:
            risk_level = 'high'
        elif combined_score >= 1.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        confidence = (ltm_analysis['confidence'] + real_time_analysis['confidence']) / 2
        
        return {
            'level': risk_level,
            'confidence': confidence,
            'combined_score': combined_score
        }
        
    def _determine_auth_requirements(self, combined_risk: Dict) -> List[str]:
        """Determine authentication requirements based on risk level"""
        risk_level = combined_risk['level']
        
        if risk_level == 'high':
            return ['biometric_verification', 'document_verification', 'in_person_verification']
        elif risk_level == 'medium':
            return ['one_time_passcode', 'device_digital_token']
        else:
            return ['knowledge_based_question', 'personal_information']

class AuthenticationOrchestrator:
    """Authentication orchestration layer"""
    
    def __init__(self, stm_agent: STMAgent, data_lake: CustomerBehaviorDataLake):
        self.stm_agent = stm_agent
        self.data_lake = data_lake
        self.auth_history: List[Dict] = []
        
    def process_authentication_request(self, transaction_context: TransactionContext) -> Dict[str, any]:
        """Process authentication request through the complete framework"""
        start_time = time.time()
        
        # Get customer profile or create new one
        customer_profile = self.data_lake.get_customer_profile(transaction_context.customer_id)
        if not customer_profile:
            customer_profile = self._create_new_customer_profile(transaction_context.customer_id)
            
        # Evaluate transaction through STM agent
        evaluation = self.stm_agent.evaluate_transaction(transaction_context)
        
        # Determine authentication method
        auth_method = self._select_authentication_method(evaluation)
        
        # Simulate authentication process
        auth_result = self._simulate_authentication(auth_method, evaluation)
        
        # Update customer profile
        self._update_customer_profile(customer_profile, transaction_context, auth_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'transaction_id': transaction_context.transaction_id,
            'risk_level': evaluation['risk_level'],
            'authentication_method': auth_method,
            'authentication_result': auth_result,
            'processing_time': processing_time,
            'confidence': evaluation['confidence']
        }
        
    def _create_new_customer_profile(self, customer_id: str) -> CustomerProfile:
        """Create new customer profile for first-time users"""
        profile = CustomerProfile(
            customer_id=customer_id,
            behavior_features={
                'total_transactions': 0,
                'avg_transaction_amount': 0.0,
                'max_transaction_amount': 0.0,
                'unique_recipients': 0,
                'login_frequency': 0,
                'device_switching_frequency': 0
            },
            digital_features={},
            risk_score=0.5,  # Default medium risk for new customers
            last_updated=datetime.now(),
            authentication_history=[]
        )
        self.data_lake.update_customer_profile(customer_id, profile)
        return profile
        
    def _select_authentication_method(self, evaluation: Dict) -> str:
        """Select appropriate authentication method based on risk level"""
        auth_requirements = evaluation['authentication_requirements']
        
        # For demonstration, return the first requirement
        return auth_requirements[0] if auth_requirements else 'knowledge_based_question'
        
    def _simulate_authentication(self, auth_method: str, evaluation: Dict) -> Dict[str, any]:
        """Simulate authentication process"""
        # Simulate success/failure based on risk level
        risk_level = evaluation['risk_level']
        
        if risk_level == 'high':
            success_rate = 0.3  # 30% success rate for high-risk transactions
        elif risk_level == 'medium':
            success_rate = 0.7  # 70% success rate for medium-risk transactions
        else:
            success_rate = 0.95  # 95% success rate for low-risk transactions
            
        is_successful = np.random.random() < success_rate
        
        return {
            'method': auth_method,
            'status': 'success' if is_successful else 'failed',
            'timestamp': datetime.now(),
            'risk_level': risk_level
        }
        
    def _update_customer_profile(self, profile: CustomerProfile, 
                               transaction_context: TransactionContext,
                               auth_result: Dict):
        """Update customer profile with new transaction and authentication data"""
        # Update authentication history
        profile.authentication_history.append(auth_result)
        
        # Update behavioral features
        if 'amount' in transaction_context.__dict__:
            current_max = profile.behavior_features.get('max_transaction_amount', 0)
            profile.behavior_features['max_transaction_amount'] = max(
                current_max, transaction_context.amount
            )
            
        profile.behavior_features['total_transactions'] += 1
        profile.last_updated = datetime.now()
        
        # Update risk score based on authentication result
        if auth_result['status'] == 'failed':
            profile.risk_score = min(1.0, profile.risk_score + 0.1)
        else:
            profile.risk_score = max(0.0, profile.risk_score - 0.05)
            
        self.data_lake.update_customer_profile(profile.customer_id, profile)

def generate_synthetic_dataset(num_customers: int = 1000, num_transactions: int = 5000) -> Tuple[List[CustomerProfile], List[TransactionContext]]:
    """Generate synthetic dataset for evaluation"""
    customers = []
    transactions = []
    
    for i in range(num_customers):
        customer_id = f"CUST_{i:06d}"
        
        # Generate behavioral features
        behavior_features = {
            'total_transactions': np.random.randint(1, 100),
            'avg_transaction_amount': np.random.uniform(100, 5000),
            'max_transaction_amount': np.random.uniform(1000, 20000),
            'unique_recipients': np.random.randint(1, 20),
            'login_frequency': np.random.randint(1, 50),
            'device_switching_frequency': np.random.randint(0, 10)
        }
        
        # Generate digital features
        digital_features = {
            'device_fingerprint': hashlib.md5(f"device_{i}".encode()).hexdigest(),
            'ip_address': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            'geolocation': f"City_{np.random.randint(1, 10)}"
        }
        
        # Create customer profile
        profile = CustomerProfile(
            customer_id=customer_id,
            behavior_features=behavior_features,
            digital_features=digital_features,
            risk_score=np.random.uniform(0.1, 0.9),
            last_updated=datetime.now() - timedelta(days=np.random.randint(1, 30)),
            authentication_history=[]
        )
        customers.append(profile)
        
        # Generate transactions for this customer
        num_customer_transactions = np.random.randint(1, 20)
        for j in range(num_customer_transactions):
            transaction = TransactionContext(
                transaction_id=f"TXN_{i:06d}_{j:04d}",
                customer_id=customer_id,
                amount=np.random.uniform(10, 15000),
                device_fingerprint=digital_features['device_fingerprint'],
                ip_address=digital_features['ip_address'],
                geolocation=digital_features['geolocation'],
                timestamp=datetime.now() - timedelta(hours=np.random.randint(1, 720)),
                transaction_type=np.random.choice(['transfer', 'payment', 'purchase']),
                recipient_info=f"RECIP_{np.random.randint(1, 1000):06d}"
            )
            transactions.append(transaction)
            
    return customers, transactions

def evaluate_framework_performance(customers: List[CustomerProfile], 
                                 transactions: List[TransactionContext]) -> Dict[str, any]:
    """Evaluate framework performance using synthetic dataset"""
    
    # Initialize framework components
    data_lake = CustomerBehaviorDataLake()
    risk_model = RiskScoringModel()
    ltm_agent = LTMAgent(data_lake, risk_model)
    stm_agent = STMAgent(ltm_agent)
    orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
    
    # Load customer profiles into data lake
    for customer in customers:
        data_lake.update_customer_profile(customer.customer_id, customer)
        
    # Process transactions and collect results
    results = []
    processing_times = []
    
    for transaction in transactions:
        result = orchestrator.process_authentication_request(transaction)
        results.append(result)
        processing_times.append(result['processing_time'])
        
    # Calculate performance metrics
    total_transactions = len(results)
    successful_auths = len([r for r in results if r['authentication_result']['status'] == 'success'])
    failed_auths = len([r for r in results if r['authentication_result']['status'] == 'failed'])
    
    # Risk level distribution
    risk_levels = [r['risk_level'] for r in results]
    high_risk_count = risk_levels.count('high')
    medium_risk_count = risk_levels.count('medium')
    low_risk_count = risk_levels.count('low')
    
    # Calculate metrics
    success_rate = successful_auths / total_transactions if total_transactions > 0 else 0
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    max_processing_time = np.max(processing_times) if processing_times else 0
    min_processing_time = np.min(processing_times) if processing_times else 0
    
    return {
        'total_transactions': total_transactions,
        'successful_authentications': successful_auths,
        'failed_authentications': failed_auths,
        'success_rate': success_rate,
        'average_processing_time': avg_processing_time,
        'max_processing_time': max_processing_time,
        'min_processing_time': min_processing_time,
        'risk_distribution': {
            'high': high_risk_count,
            'medium': medium_risk_count,
            'low': low_risk_count
        },
        'risk_distribution_percentage': {
            'high': (high_risk_count / total_transactions) * 100 if total_transactions > 0 else 0,
            'medium': (medium_risk_count / total_transactions) * 100 if total_transactions > 0 else 0,
            'low': (low_risk_count / total_transactions) * 100 if total_transactions > 0 else 0
        }
    }

def main():
    """Main function to demonstrate the PoC implementation"""
    print("=" * 80)
    print("RISK-BASED FLEXIBLE AUTHENTICATION FRAMEWORK - PROOF OF CONCEPT")
    print("=" * 80)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    customers, transactions = generate_synthetic_dataset(num_customers=500, num_transactions=2000)
    print(f"   Generated {len(customers)} customers and {len(transactions)} transactions")
    
    # Evaluate framework performance
    print("\n2. Evaluating framework performance...")
    performance_metrics = evaluate_framework_performance(customers, transactions)
    
    # Display results
    print("\n3. PERFORMANCE METRICS:")
    print("-" * 50)
    print(f"Total Transactions Processed: {performance_metrics['total_transactions']}")
    print(f"Successful Authentications: {performance_metrics['successful_authentications']}")
    print(f"Failed Authentications: {performance_metrics['failed_authentications']}")
    print(f"Success Rate: {performance_metrics['success_rate']:.2%}")
    print(f"Average Processing Time: {performance_metrics['average_processing_time']:.4f} seconds")
    print(f"Max Processing Time: {performance_metrics['max_processing_time']:.4f} seconds")
    print(f"Min Processing Time: {performance_metrics['min_processing_time']:.4f} seconds")
    
    print("\nRisk Level Distribution:")
    for risk_level, count in performance_metrics['risk_distribution'].items():
        percentage = performance_metrics['risk_distribution_percentage'][risk_level]
        print(f"  {risk_level.upper()}: {count} transactions ({percentage:.1f}%)")
    
    # Demonstrate individual transaction processing
    print("\n4. SAMPLE TRANSACTION PROCESSING:")
    print("-" * 50)
    
    # Initialize framework for demonstration
    data_lake = CustomerBehaviorDataLake()
    risk_model = RiskScoringModel()
    ltm_agent = LTMAgent(data_lake, risk_model)
    stm_agent = STMAgent(ltm_agent)
    orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
    
    # Load a sample customer
    sample_customer = customers[0]
    data_lake.update_customer_profile(sample_customer.customer_id, sample_customer)
    
    # Process a sample transaction
    sample_transaction = transactions[0]
    result = orchestrator.process_authentication_request(sample_transaction)
    
    print(f"Transaction ID: {result['transaction_id']}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Authentication Method: {result['authentication_method']}")
    print(f"Authentication Result: {result['authentication_result']['status'].upper()}")
    print(f"Processing Time: {result['processing_time']:.4f} seconds")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\n" + "=" * 80)
    print("PROOF OF CONCEPT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main() 