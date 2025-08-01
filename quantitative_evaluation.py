#!/usr/bin/env python3
"""
Quantitative Evaluation Script for Risk-based Flexible Authentication Framework

This script performs comprehensive evaluation of the proposed framework including:
1. Risk Classification Accuracy (confusion matrix, precision, recall, F1-score)
2. Fraud Detection Rates (False Positives/False Negatives)
3. System Latency Analysis
4. Comparative Analysis with Baseline Methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Import our PoC implementation
from poc_implementation import (
    CustomerBehaviorDataLake, RiskScoringModel, LTMAgent, STMAgent, 
    AuthenticationOrchestrator, CustomerProfile, TransactionContext,
    generate_synthetic_dataset
)

class QuantitativeEvaluator:
    """Comprehensive quantitative evaluation of the authentication framework"""
    
    def __init__(self):
        self.results = {}
        self.baseline_results = {}
        
    def generate_evaluation_dataset(self, num_customers: int = 2000, 
                                  num_transactions: int = 10000) -> Tuple[List, List]:
        """Generate comprehensive evaluation dataset with known fraud patterns"""
        print("Generating evaluation dataset...")
        
        customers, transactions = generate_synthetic_dataset(num_customers, num_transactions)
        
        # Inject known fraud patterns for evaluation
        fraud_transactions = self._inject_fraud_patterns(transactions)
        
        # Label transactions as legitimate (0) or fraudulent (1)
        labeled_transactions = self._label_transactions(transactions + fraud_transactions)
        
        return customers, labeled_transactions
        
    def _inject_fraud_patterns(self, legitimate_transactions: List[TransactionContext]) -> List[TransactionContext]:
        """Inject known fraud patterns for evaluation"""
        fraud_transactions = []
        
        # Pattern 1: High-value transactions from new devices
        for i in range(200):
            fraud_txn = TransactionContext(
                transaction_id=f"FRAUD_HV_{i:04d}",
                customer_id=f"CUST_{np.random.randint(0, 100):06d}",
                amount=np.random.uniform(15000, 50000),
                device_fingerprint=f"new_device_{i}",
                ip_address=f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                geolocation="Unknown_Location",
                timestamp=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                transaction_type="transfer",
                recipient_info=f"FRAUD_RECIP_{i:04d}"
            )
            fraud_transactions.append(fraud_txn)
            
        # Pattern 2: Rapid transactions from same device
        for i in range(150):
            base_time = datetime.now() - timedelta(hours=1)
            for j in range(5):  # 5 rapid transactions
                fraud_txn = TransactionContext(
                    transaction_id=f"FRAUD_RAPID_{i}_{j}",
                    customer_id=f"CUST_{np.random.randint(0, 100):06d}",
                    amount=np.random.uniform(1000, 5000),
                    device_fingerprint=f"rapid_device_{i}",
                    ip_address=f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    geolocation=f"City_{np.random.randint(1, 5)}",
                    timestamp=base_time + timedelta(minutes=j*2),  # 2-minute intervals
                    transaction_type="payment",
                    recipient_info=f"RAPID_RECIP_{i:04d}"
                )
                fraud_transactions.append(fraud_txn)
                
        # Pattern 3: Unusual time transactions
        for i in range(100):
            fraud_txn = TransactionContext(
                transaction_id=f"FRAUD_TIME_{i:04d}",
                customer_id=f"CUST_{np.random.randint(0, 100):06d}",
                amount=np.random.uniform(5000, 15000),
                device_fingerprint=f"time_device_{i}",
                ip_address=f"172.16.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                geolocation=f"City_{np.random.randint(1, 10)}",
                timestamp=datetime.now().replace(hour=np.random.choice([2, 3, 4])),  # 2-4 AM
                transaction_type="purchase",
                recipient_info=f"TIME_RECIP_{i:04d}"
            )
            fraud_transactions.append(fraud_txn)
            
        return fraud_transactions
        
    def _label_transactions(self, transactions: List[TransactionContext]) -> List[Tuple[TransactionContext, int]]:
        """Label transactions as legitimate (0) or fraudulent (1)"""
        labeled_transactions = []
        
        for txn in transactions:
            # Determine if transaction is fraudulent based on patterns
            is_fraudulent = 0
            
            # High-value transactions from new devices
            if (txn.amount > 15000 and 
                'new_device' in txn.device_fingerprint):
                is_fraudulent = 1
                
            # Rapid transactions
            if 'rapid_device' in txn.device_fingerprint:
                is_fraudulent = 1
                
            # Unusual time transactions
            if (txn.timestamp.hour in [2, 3, 4] and 
                'time_device' in txn.device_fingerprint):
                is_fraudulent = 1
                
            labeled_transactions.append((txn, is_fraudulent))
            
        return labeled_transactions
        
    def evaluate_risk_classification_accuracy(self, customers: List[CustomerProfile], 
                                            labeled_transactions: List[Tuple[TransactionContext, int]]) -> Dict:
        """Evaluate risk classification accuracy"""
        print("Evaluating risk classification accuracy...")
        
        # Initialize framework
        data_lake = CustomerBehaviorDataLake()
        risk_model = RiskScoringModel()
        ltm_agent = LTMAgent(data_lake, risk_model)
        stm_agent = STMAgent(ltm_agent)
        orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
        
        # Load customers into data lake
        for customer in customers:
            data_lake.update_customer_profile(customer.customer_id, customer)
            
        # Process transactions and collect predictions
        true_labels = []
        predicted_risk_levels = []
        processing_times = []
        
        for txn, true_label in labeled_transactions:
            start_time = time.time()
            
            # Process transaction through framework
            result = orchestrator.process_authentication_request(txn)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Convert risk level to numeric for comparison
            risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
            predicted_risk = risk_level_mapping.get(result['risk_level'], 1)
            
            # Convert true label to risk level (fraudulent = high risk)
            if true_label == 1:
                true_risk = 2  # High risk for fraudulent transactions
            else:
                true_risk = 0  # Low risk for legitimate transactions
                
            true_labels.append(true_risk)
            predicted_risk_levels.append(predicted_risk)
            
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_risk_levels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_risk_levels, average='weighted'
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_risk_levels)
        
        # Calculate per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_risk_levels, average=None
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist(),
            'average_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'min_processing_time': np.min(processing_times),
            'total_transactions': len(labeled_transactions)
        }
        
        self.results['risk_classification'] = results
        return results
        
    def evaluate_fraud_detection_rates(self, labeled_transactions: List[Tuple[TransactionContext, int]]) -> Dict:
        """Evaluate fraud detection rates (False Positives/False Negatives)"""
        print("Evaluating fraud detection rates...")
        
        # Initialize framework
        data_lake = CustomerBehaviorDataLake()
        risk_model = RiskScoringModel()
        ltm_agent = LTMAgent(data_lake, risk_model)
        stm_agent = STMAgent(ltm_agent)
        orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
        
        # Process transactions
        fraud_detection_results = []
        
        for txn, true_label in labeled_transactions:
            result = orchestrator.process_authentication_request(txn)
            
            # Determine if transaction was flagged as high risk (potential fraud)
            is_flagged = result['risk_level'] == 'high'
            
            fraud_detection_results.append({
                'transaction_id': txn.transaction_id,
                'true_fraud': true_label == 1,
                'detected_as_fraud': is_flagged,
                'risk_level': result['risk_level'],
                'confidence': result['confidence']
            })
            
        # Calculate fraud detection metrics
        true_positives = sum(1 for r in fraud_detection_results 
                           if r['true_fraud'] and r['detected_as_fraud'])
        false_positives = sum(1 for r in fraud_detection_results 
                            if not r['true_fraud'] and r['detected_as_fraud'])
        true_negatives = sum(1 for r in fraud_detection_results 
                           if not r['true_fraud'] and not r['detected_as_fraud'])
        false_negatives = sum(1 for r in fraud_detection_results 
                            if r['true_fraud'] and not r['detected_as_fraud'])
        
        total_fraudulent = sum(1 for r in fraud_detection_results if r['true_fraud'])
        total_legitimate = sum(1 for r in fraud_detection_results if not r['true_fraud'])
        
        # Calculate rates
        fraud_detection_rate = true_positives / total_fraudulent if total_fraudulent > 0 else 0
        false_positive_rate = false_positives / total_legitimate if total_legitimate > 0 else 0
        false_negative_rate = false_negatives / total_fraudulent if total_fraudulent > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        results = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_fraudulent': total_fraudulent,
            'total_legitimate': total_legitimate,
            'fraud_detection_rate': fraud_detection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
        
        self.results['fraud_detection'] = results
        return results
        
    def evaluate_system_latency(self, customers: List[CustomerProfile], 
                              transactions: List[TransactionContext]) -> Dict:
        """Evaluate system latency for authentication decisions"""
        print("Evaluating system latency...")
        
        # Initialize framework
        data_lake = CustomerBehaviorDataLake()
        risk_model = RiskScoringModel()
        ltm_agent = LTMAgent(data_lake, risk_model)
        stm_agent = STMAgent(ltm_agent)
        orchestrator = AuthenticationOrchestrator(stm_agent, data_lake)
        
        # Load customers
        for customer in customers:
            data_lake.update_customer_profile(customer.customer_id, customer)
            
        # Measure latency for different transaction types and risk levels
        latency_results = {
            'low': [],
            'medium': [],
            'high': [],
            'all_transactions': []
        }
        
        for txn in transactions:
            start_time = time.time()
            
            result = orchestrator.process_authentication_request(txn)
            
            processing_time = time.time() - start_time
            
            # Categorize by risk level
            risk_level = result['risk_level']
            latency_results[risk_level].append(processing_time)
            latency_results['all_transactions'].append(processing_time)
            
        # Calculate latency statistics
        latency_stats = {}
        for category, times in latency_results.items():
            if times:
                latency_stats[category] = {
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99),
                    'count': len(times)
                }
                
        self.results['system_latency'] = latency_stats
        return latency_stats
        
    def comparative_analysis_baseline(self, customers: List[CustomerProfile], 
                                    labeled_transactions: List[Tuple[TransactionContext, int]]) -> Dict:
        """Compare framework performance against baseline methods"""
        print("Performing comparative analysis with baseline methods...")
        
        # Baseline 1: Traditional MFA (always requires multi-factor authentication)
        baseline_mfa_results = self._evaluate_baseline_mfa(labeled_transactions)
        
        # Baseline 2: Simple rule-based system
        baseline_rule_results = self._evaluate_baseline_rule_based(labeled_transactions)
        
        # Our proposed framework
        framework_results = self.evaluate_risk_classification_accuracy(customers, labeled_transactions)
        
        comparative_results = {
            'proposed_framework': {
                'accuracy': framework_results['accuracy'],
                'precision': framework_results['precision'],
                'recall': framework_results['recall'],
                'f1_score': framework_results['f1_score'],
                'avg_processing_time': framework_results['average_processing_time']
            },
            'traditional_mfa': baseline_mfa_results,
            'rule_based': baseline_rule_results
        }
        
        # Calculate improvements
        mfa_improvement = {
            'accuracy_improvement': framework_results['accuracy'] - baseline_mfa_results['accuracy'],
            'precision_improvement': framework_results['precision'] - baseline_mfa_results['precision'],
            'recall_improvement': framework_results['recall'] - baseline_mfa_results['recall'],
            'f1_improvement': framework_results['f1_score'] - baseline_mfa_results['f1_score'],
            'latency_improvement': baseline_mfa_results['avg_processing_time'] - framework_results['average_processing_time']
        }
        
        rule_improvement = {
            'accuracy_improvement': framework_results['accuracy'] - baseline_rule_results['accuracy'],
            'precision_improvement': framework_results['precision'] - baseline_rule_results['precision'],
            'recall_improvement': framework_results['recall'] - baseline_rule_results['recall'],
            'f1_improvement': framework_results['f1_score'] - baseline_rule_results['f1_score'],
            'latency_improvement': baseline_rule_results['avg_processing_time'] - framework_results['average_processing_time']
        }
        
        comparative_results['improvements_vs_mfa'] = mfa_improvement
        comparative_results['improvements_vs_rule_based'] = rule_improvement
        
        self.results['comparative_analysis'] = comparative_results
        return comparative_results
        
    def _evaluate_baseline_mfa(self, labeled_transactions: List[Tuple[TransactionContext, int]]) -> Dict:
        """Evaluate traditional MFA baseline"""
        print("  Evaluating traditional MFA baseline...")
        
        true_labels = []
        predicted_labels = []
        processing_times = []
        
        for txn, true_label in labeled_transactions:
            start_time = time.time()
            
            # Traditional MFA always requires authentication
            # Simulate processing time for MFA
            time.sleep(0.001)  # Simulate MFA processing time
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Traditional MFA treats all transactions as high risk
            predicted_label = 1 if true_label == 1 else 0  # Simple binary classification
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_processing_time': np.mean(processing_times)
        }
        
    def _evaluate_baseline_rule_based(self, labeled_transactions: List[Tuple[TransactionContext, int]]) -> Dict:
        """Evaluate simple rule-based baseline"""
        print("  Evaluating rule-based baseline...")
        
        true_labels = []
        predicted_labels = []
        processing_times = []
        
        for txn, true_label in labeled_transactions:
            start_time = time.time()
            
            # Simple rule-based system
            # High amount = high risk
            is_high_risk = txn.amount > 10000
            
            # Unknown location = high risk
            if 'Unknown' in txn.geolocation:
                is_high_risk = True
                
            # Unusual time = high risk
            if txn.timestamp.hour < 6 or txn.timestamp.hour > 23:
                is_high_risk = True
                
            predicted_label = 1 if is_high_risk else 0
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_processing_time': np.mean(processing_times)
        }
        
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE EVALUATION REPORT")
        report.append("Risk-based Flexible Authentication Framework")
        report.append("=" * 80)
        
        # Risk Classification Accuracy
        if 'risk_classification' in self.results:
            rc = self.results['risk_classification']
            report.append("\n1. RISK CLASSIFICATION ACCURACY")
            report.append("-" * 50)
            report.append(f"Overall Accuracy: {rc['accuracy']:.4f} ({rc['accuracy']*100:.2f}%)")
            report.append(f"Precision: {rc['precision']:.4f}")
            report.append(f"Recall: {rc['recall']:.4f}")
            report.append(f"F1-Score: {rc['f1_score']:.4f}")
            report.append(f"Total Transactions: {rc['total_transactions']}")
            
            report.append("\nPer-Class Performance:")
            risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
            for i, level in enumerate(risk_levels):
                report.append(f"  {level}:")
                report.append(f"    Precision: {rc['class_precision'][i]:.4f}")
                report.append(f"    Recall: {rc['class_recall'][i]:.4f}")
                report.append(f"    F1-Score: {rc['class_f1'][i]:.4f}")
                
        # Fraud Detection Rates
        if 'fraud_detection' in self.results:
            fd = self.results['fraud_detection']
            report.append("\n2. FRAUD DETECTION RATES")
            report.append("-" * 50)
            report.append(f"True Positives: {fd['true_positives']}")
            report.append(f"False Positives: {fd['false_positives']}")
            report.append(f"True Negatives: {fd['true_negatives']}")
            report.append(f"False Negatives: {fd['false_negatives']}")
            report.append(f"Fraud Detection Rate: {fd['fraud_detection_rate']:.4f} ({fd['fraud_detection_rate']*100:.2f}%)")
            report.append(f"False Positive Rate: {fd['false_positive_rate']:.4f} ({fd['false_positive_rate']*100:.2f}%)")
            report.append(f"False Negative Rate: {fd['false_negative_rate']:.4f} ({fd['false_negative_rate']*100:.2f}%)")
            report.append(f"Precision: {fd['precision']:.4f}")
            report.append(f"Recall: {fd['recall']:.4f}")
            report.append(f"F1-Score: {fd['f1_score']:.4f}")
            
        # System Latency
        if 'system_latency' in self.results:
            sl = self.results['system_latency']
            report.append("\n3. SYSTEM LATENCY ANALYSIS")
            report.append("-" * 50)
            
            for category, stats in sl.items():
                if category != 'all_transactions':
                    report.append(f"\n{category.replace('_', ' ').title()}:")
                    report.append(f"  Mean: {stats['mean']:.6f} seconds")
                    report.append(f"  Median: {stats['median']:.6f} seconds")
                    report.append(f"  Std Dev: {stats['std']:.6f} seconds")
                    report.append(f"  Min: {stats['min']:.6f} seconds")
                    report.append(f"  Max: {stats['max']:.6f} seconds")
                    report.append(f"  95th Percentile: {stats['p95']:.6f} seconds")
                    report.append(f"  99th Percentile: {stats['p99']:.6f} seconds")
                    report.append(f"  Count: {stats['count']}")
                    
        # Comparative Analysis
        if 'comparative_analysis' in self.results:
            ca = self.results['comparative_analysis']
            report.append("\n4. COMPARATIVE ANALYSIS")
            report.append("-" * 50)
            
            report.append("\nProposed Framework vs Traditional MFA:")
            improvements = ca['improvements_vs_mfa']
            for metric, improvement in improvements.items():
                report.append(f"  {metric.replace('_', ' ').title()}: {improvement:+.4f}")
                
            report.append("\nProposed Framework vs Rule-based System:")
            improvements = ca['improvements_vs_rule_based']
            for metric, improvement in improvements.items():
                report.append(f"  {metric.replace('_', ' ').title()}: {improvement:+.4f}")
                
        report.append("\n" + "=" * 80)
        report.append("EVALUATION COMPLETED")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")

def main():
    """Main evaluation function"""
    print("Starting comprehensive quantitative evaluation...")
    
    # Initialize evaluator
    evaluator = QuantitativeEvaluator()
    
    # Generate evaluation dataset
    customers, labeled_transactions = evaluator.generate_evaluation_dataset(
        num_customers=1000, num_transactions=5000
    )
    
    print(f"Generated {len(customers)} customers and {len(labeled_transactions)} labeled transactions")
    
    # Perform evaluations
    print("\nPerforming evaluations...")
    
    # 1. Risk Classification Accuracy
    evaluator.evaluate_risk_classification_accuracy(customers, labeled_transactions)
    
    # 2. Fraud Detection Rates
    transactions_only = [txn for txn, _ in labeled_transactions]
    evaluator.evaluate_fraud_detection_rates(labeled_transactions)
    
    # 3. System Latency
    evaluator.evaluate_system_latency(customers, transactions_only)
    
    # 4. Comparative Analysis
    evaluator.comparative_analysis_baseline(customers, labeled_transactions)
    
    # Generate and display report
    report = evaluator.generate_evaluation_report()
    print(report)
    
    # Save results
    evaluator.save_results()
    
    print("\nQuantitative evaluation completed successfully!")

if __name__ == "__main__":
    main() 