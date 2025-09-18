#!/usr/bin/env python3
"""
Statistical Process Control (SPC) Analysis
Implements control charts and quality metrics for manufacturing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import statistics
import json

logger = logging.getLogger(__name__)

class SPCAnalyzer:
    """Statistical Process Control analyzer for quality metrics"""
    
    def __init__(self, window_size: int = 100, confidence_level: float = 0.95):
        """
        Initialize SPC analyzer
        
        Args:
            window_size: Size of rolling window for calculations
            confidence_level: Confidence level for control limits
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.data_buffer = deque(maxlen=window_size)
        self.control_limits = {}
        self.process_capability = {}
        self.trend_data = {
            'timestamps': deque(maxlen=window_size),
            'defect_counts': deque(maxlen=window_size),
            'defect_rates': deque(maxlen=window_size),
            'confidence_scores': deque(maxlen=window_size)
        }
        
        # SPC rules for out-of-control detection
        self.spc_rules = {
            'rule1': 'Point beyond control limits',
            'rule2': '9 consecutive points on same side of centerline',
            'rule3': '6 consecutive increasing or decreasing points',
            'rule4': '14 alternating up and down points',
            'rule5': '2 out of 3 consecutive points beyond 2-sigma',
            'rule6': '4 out of 5 consecutive points beyond 1-sigma',
            'rule7': '15 consecutive points within 1-sigma',
            'rule8': '8 consecutive points beyond 1-sigma'
        }
    
    def analyze(self, prediction_results: Dict) -> Dict:
        """
        Analyze prediction results for SPC metrics
        
        Args:
            prediction_results: Results from quality control system
            
        Returns:
            SPC analysis results
        """
        try:
            # Extract metrics from prediction results
            metrics = self._extract_metrics(prediction_results)
            
            # Add to data buffer
            self._update_data_buffer(metrics)
            
            # Calculate control limits
            control_limits = self._calculate_control_limits()
            
            # Check for out-of-control conditions
            control_status = self._check_control_status(metrics, control_limits)
            
            # Calculate process capability
            capability_metrics = self._calculate_process_capability()
            
            # Generate alerts if necessary
            alerts = self._generate_alerts(control_status, metrics)
            
            # Create SPC chart data
            chart_data = self._prepare_chart_data()
            
            spc_results = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': metrics,
                'control_limits': control_limits,
                'control_status': control_status,
                'process_capability': capability_metrics,
                'alerts': alerts,
                'chart_data': chart_data,
                'trend_analysis': self._analyze_trends(),
                'recommendations': self._generate_recommendations(control_status, capability_metrics)
            }
            
            return spc_results
            
        except Exception as e:
            logger.error(f"Error in SPC analysis: {str(e)}")
            return {'error': str(e)}
    
    def _extract_metrics(self, prediction_results: Dict) -> Dict:
        """Extract relevant metrics from prediction results"""
        metrics = {
            'timestamp': datetime.now(),
            'defect_count': 0,
            'defect_rate': 0.0,
            'avg_confidence': 0.0,
            'critical_defects': 0,
            'major_defects': 0,
            'minor_defects': 0,
            'total_area_affected': 0.0
        }
        
        detections = prediction_results.get('detections', [])
        
        if detections:
            metrics['defect_count'] = len(detections)
            metrics['defect_rate'] = len(detections)  # Could be normalized by inspection area
            
            # Calculate confidence statistics
            confidences = [d.get('ensemble_confidence', d.get('confidence', 0.0)) for d in detections]
            metrics['avg_confidence'] = np.mean(confidences) if confidences else 0.0
            
            # Count by severity
            for detection in detections:
                severity = detection.get('final_severity', detection.get('severity', 'minor'))
                if severity == 'critical':
                    metrics['critical_defects'] += 1
                elif severity == 'major':
                    metrics['major_defects'] += 1
                else:
                    metrics['minor_defects'] += 1
                
                # Sum affected area
                bbox = detection.get('bbox', {})
                area = bbox.get('width', 0) * bbox.get('height', 0)
                metrics['total_area_affected'] += area
        
        # Calculate derived metrics
        metrics['critical_rate'] = metrics['critical_defects'] / max(1, metrics['defect_count'])
        metrics['major_rate'] = metrics['major_defects'] / max(1, metrics['defect_count'])
        
        return metrics
    
    def _update_data_buffer(self, metrics: Dict):
        """Update rolling data buffer"""
        self.data_buffer.append(metrics)
        
        # Update trend data
        self.trend_data['timestamps'].append(metrics['timestamp'])
        self.trend_data['defect_counts'].append(metrics['defect_count'])
        self.trend_data['defect_rates'].append(metrics['defect_rate'])
        self.trend_data['confidence_scores'].append(metrics['avg_confidence'])
    
    def _calculate_control_limits(self) -> Dict:
        """Calculate control limits for various metrics"""
        if len(self.data_buffer) < 10:  # Need minimum data points
            return {}
        
        control_limits = {}
        
        # Extract time series data
        defect_counts = [d['defect_count'] for d in self.data_buffer]
        defect_rates = [d['defect_rate'] for d in self.data_buffer]
        confidences = [d['avg_confidence'] for d in self.data_buffer]
        
        # Calculate limits for defect count (c-chart)
        control_limits['defect_count'] = self._calculate_c_chart_limits(defect_counts)
        
        # Calculate limits for defect rate (u-chart)
        control_limits['defect_rate'] = self._calculate_u_chart_limits(defect_rates)
        
        # Calculate limits for confidence (X-chart)
        control_limits['confidence'] = self._calculate_x_chart_limits(confidences)
        
        return control_limits
    
    def _calculate_c_chart_limits(self, data: List[float]) -> Dict:
        """Calculate control limits for c-chart (count data)"""
        c_bar = np.mean(data)
        
        return {
            'center_line': c_bar,
            'upper_control_limit': c_bar + 3 * np.sqrt(c_bar),
            'lower_control_limit': max(0, c_bar - 3 * np.sqrt(c_bar)),
            'upper_warning_limit': c_bar + 2 * np.sqrt(c_bar),
            'lower_warning_limit': max(0, c_bar - 2 * np.sqrt(c_bar)),
            'chart_type': 'c-chart'
        }
    
    def _calculate_u_chart_limits(self, data: List[float]) -> Dict:
        """Calculate control limits for u-chart (rate data)"""
        u_bar = np.mean(data)
        n = 1  # Sample size (could be configurable)
        
        return {
            'center_line': u_bar,
            'upper_control_limit': u_bar + 3 * np.sqrt(u_bar / n),
            'lower_control_limit': max(0, u_bar - 3 * np.sqrt(u_bar / n)),
            'upper_warning_limit': u_bar + 2 * np.sqrt(u_bar / n),
            'lower_warning_limit': max(0, u_bar - 2 * np.sqrt(u_bar / n)),
            'chart_type': 'u-chart'
        }
    
    def _calculate_x_chart_limits(self, data: List[float]) -> Dict:
        """Calculate control limits for X-chart (individual values)"""
        x_bar = np.mean(data)
        
        # Calculate moving range
        moving_ranges = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
        mr_bar = np.mean(moving_ranges) if moving_ranges else 0
        
        # Constants for individuals chart
        d2 = 1.128  # for n=2 (moving range)
        
        return {
            'center_line': x_bar,
            'upper_control_limit': x_bar + 3 * (mr_bar / d2),
            'lower_control_limit': x_bar - 3 * (mr_bar / d2),
            'upper_warning_limit': x_bar + 2 * (mr_bar / d2),
            'lower_warning_limit': x_bar - 2 * (mr_bar / d2),
            'chart_type': 'X-chart'
        }
    
    def _check_control_status(self, current_metrics: Dict, control_limits: Dict) -> Dict:
        """Check if process is in control"""
        control_status = {
            'in_control': True,
            'violations': [],
            'warnings': []
        }
        
        # Check each metric against control limits
        for metric_name, limits in control_limits.items():
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Check for violations (beyond control limits)
            if (current_value > limits['upper_control_limit'] or 
                current_value < limits['lower_control_limit']):
                control_status['in_control'] = False
                control_status['violations'].append({
                    'metric': metric_name,
                    'value': current_value,
                    'limit_violated': 'upper' if current_value > limits['upper_control_limit'] else 'lower',
                    'severity': 'critical'
                })
            
            # Check for warnings (beyond warning limits)
            elif (current_value > limits['upper_warning_limit'] or 
                  current_value < limits['lower_warning_limit']):
                control_status['warnings'].append({
                    'metric': metric_name,
                    'value': current_value,
                    'limit_violated': 'upper' if current_value > limits['upper_warning_limit'] else 'lower',
                    'severity': 'warning'
                })
        
        # Check SPC rules
        rule_violations = self._check_spc_rules()
        control_status['rule_violations'] = rule_violations
        
        if rule_violations:
            control_status['in_control'] = False
        
        return control_status
    
    def _check_spc_rules(self) -> List[Dict]:
        """Check various SPC rules for out-of-control patterns"""
        violations = []
        
        if len(self.data_buffer) < 9:  # Need minimum data for rule checking
            return violations
        
        # Extract recent data points
        recent_data = list(self.data_buffer)[-20:]  # Last 20 points
        defect_counts = [d['defect_count'] for d in recent_data]
        
        # Calculate centerline for recent data
        centerline = np.mean(defect_counts)
        
        # Rule 2: 9 consecutive points on same side of centerline
        if len(defect_counts) >= 9:
            for i in range(len(defect_counts) - 8):
                sequence = defect_counts[i:i+9]
                if all(x > centerline for x in sequence) or all(x < centerline for x in sequence):
                    violations.append({
                        'rule': 'rule2',
                        'description': self.spc_rules['rule2'],
                        'severity': 'major'
                    })
                    break
        
        # Rule 3: 6 consecutive increasing or decreasing points
        if len(defect_counts) >= 6:
            for i in range(len(defect_counts) - 5):
                sequence = defect_counts[i:i+6]
                if (all(sequence[j] < sequence[j+1] for j in range(5)) or
                    all(sequence[j] > sequence[j+1] for j in range(5))):
                    violations.append({
                        'rule': 'rule3',
                        'description': self.spc_rules['rule3'],
                        'severity': 'major'
                    })
                    break
        
        return violations
    
    def _calculate_process_capability(self) -> Dict:
        """Calculate process capability indices"""
        if len(self.data_buffer) < 30:  # Need sufficient data
            return {'insufficient_data': True}
        
        # Extract defect rate data
        defect_rates = [d['defect_rate'] for d in self.data_buffer]
        
        # Calculate basic statistics
        mean_rate = np.mean(defect_rates)
        std_rate = np.std(defect_rates, ddof=1)
        
        # Define specification limits (configurable)
        upper_spec_limit = 5.0  # Example: max 5 defects per unit
        lower_spec_limit = 0.0
        target = 0.5  # Target defect rate
        
        # Calculate capability indices
        capability = {}
        
        if std_rate > 0:
            # Cp: Process capability (precision)
            cp = (upper_spec_limit - lower_spec_limit) / (6 * std_rate)
            
            # Cpk: Process capability (precision + accuracy)
            cpu = (upper_spec_limit - mean_rate) / (3 * std_rate)
            cpl = (mean_rate - lower_spec_limit) / (3 * std_rate)
            cpk = min(cpu, cpl)
            
            # Cpm: Process capability with target
            cpm = (upper_spec_limit - lower_spec_limit) / (6 * np.sqrt(std_rate**2 + (mean_rate - target)**2))
            
            capability.update({
                'cp': cp,
                'cpk': cpk,
                'cpm': cpm,
                'cpu': cpu,
                'cpl': cpl,
                'mean': mean_rate,
                'std': std_rate,
                'target': target,
                'specification_limits': {
                    'upper': upper_spec_limit,
                    'lower': lower_spec_limit
                }
            })
            
            # Interpret capability
            capability['interpretation'] = self._interpret_capability(cpk)
        
        return capability
    
    def _interpret_capability(self, cpk: float) -> str:
        """Interpret process capability index"""
        if cpk >= 2.0:
            return "Excellent - 6 sigma process"
        elif cpk >= 1.67:
            return "Very Good - 5 sigma process"
        elif cpk >= 1.33:
            return "Good - 4 sigma process"
        elif cpk >= 1.0:
            return "Adequate - 3 sigma process"
        elif cpk >= 0.67:
            return "Poor - Process improvement needed"
        else:
            return "Unacceptable - Immediate action required"
    
    def _generate_alerts(self, control_status: Dict, metrics: Dict) -> List[Dict]:
        """Generate alerts based on control status"""
        alerts = []
        
        # Control limit violations
        for violation in control_status.get('violations', []):
            alerts.append({
                'type': 'control_violation',
                'severity': 'critical',
                'message': f"Control limit violation: {violation['metric']} = {violation['value']:.2f}",
                'timestamp': datetime.now().isoformat(),
                'action_required': True
            })
        
        # Warning limit violations
        for warning in control_status.get('warnings', []):
            alerts.append({
                'type': 'warning_limit',
                'severity': 'warning',
                'message': f"Warning limit exceeded: {warning['metric']} = {warning['value']:.2f}",
                'timestamp': datetime.now().isoformat(),
                'action_required': False
            })
        
        # High defect rate alert
        if metrics['defect_rate'] > 3.0:  # Configurable threshold
            alerts.append({
                'type': 'high_defect_rate',
                'severity': 'major',
                'message': f"High defect rate detected: {metrics['defect_rate']:.2f}",
                'timestamp': datetime.now().isoformat(),
                'action_required': True
            })
        
        # Critical defects alert
        if metrics['critical_defects'] > 0:
            alerts.append({
                'type': 'critical_defects',
                'severity': 'critical',
                'message': f"Critical defects detected: {metrics['critical_defects']}",
                'timestamp': datetime.now().isoformat(),
                'action_required': True
            })
        
        return alerts
    
    def _prepare_chart_data(self) -> Dict:
        """Prepare data for SPC charts"""
        if len(self.trend_data['timestamps']) == 0:
            return {}
        
        chart_data = {
            'timestamps': [ts.isoformat() for ts in self.trend_data['timestamps']],
            'defect_counts': list(self.trend_data['defect_counts']),
            'defect_rates': list(self.trend_data['defect_rates']),
            'confidence_scores': list(self.trend_data['confidence_scores'])
        }
        
        # Add control limits if available
        if self.control_limits:
            chart_data['control_limits'] = self.control_limits
        
        return chart_data
    
    def _analyze_trends(self) -> Dict:
        """Analyze trends in the data"""
        if len(self.data_buffer) < 10:
            return {'insufficient_data': True}
        
        # Extract recent data
        recent_defect_counts = list(self.trend_data['defect_counts'])[-20:]
        recent_timestamps = list(self.trend_data['timestamps'])[-20:]
        
        # Calculate trend
        if len(recent_defect_counts) >= 5:
            x = np.arange(len(recent_defect_counts))
            slope, intercept = np.polyfit(x, recent_defect_counts, 1)
            
            trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            
            return {
                'trend_direction': trend_direction,
                'slope': float(slope),
                'recent_average': np.mean(recent_defect_counts[-5:]),
                'overall_average': np.mean(recent_defect_counts),
                'volatility': np.std(recent_defect_counts)
            }
        
        return {}
    
    def _generate_recommendations(self, control_status: Dict, capability: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Control status recommendations
        if not control_status.get('in_control', True):
            recommendations.append("Process is out of control - investigate special causes")
            recommendations.append("Review recent process changes or environmental factors")
        
        # Capability recommendations
        if not capability.get('insufficient_data', True):
            cpk = capability.get('cpk', 0)
            if cpk < 1.0:
                recommendations.append("Process capability is inadequate - consider process improvement")
            elif cpk < 1.33:
                recommendations.append("Process capability is marginal - monitor closely")
        
        # High defect rate recommendations
        recent_avg = np.mean([d['defect_rate'] for d in list(self.data_buffer)[-5:]]) if len(self.data_buffer) >= 5 else 0
        if recent_avg > 2.0:
            recommendations.append("High defect rate detected - review quality procedures")
            recommendations.append("Consider additional operator training or equipment maintenance")
        
        # Trend-based recommendations
        trend_analysis = self._analyze_trends()
        if trend_analysis.get('trend_direction') == 'increasing':
            recommendations.append("Increasing defect trend detected - preventive action recommended")
        
        return recommendations
    
    def export_spc_report(self, filepath: str) -> bool:
        """Export SPC analysis report"""
        try:
            if len(self.data_buffer) == 0:
                logger.warning("No data available for SPC report")
                return False
            
            # Prepare report data
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_samples': len(self.data_buffer),
                    'time_period': {
                        'start': list(self.data_buffer)[0]['timestamp'].isoformat(),
                        'end': list(self.data_buffer)[-1]['timestamp'].isoformat()
                    }
                },
                'control_limits': self.control_limits,
                'process_capability': self.process_capability,
                'recent_data': [
                    {
                        'timestamp': d['timestamp'].isoformat(),
                        'defect_count': d['defect_count'],
                        'defect_rate': d['defect_rate'],
                        'avg_confidence': d['avg_confidence']
                    }
                    for d in list(self.data_buffer)[-50:]  # Last 50 samples
                ]
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"SPC report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting SPC report: {str(e)}")
            return False
    
    def reset_data(self):
        """Reset all accumulated data"""
        self.data_buffer.clear()
        self.control_limits.clear()
        self.process_capability.clear()
        for key in self.trend_data:
            self.trend_data[key].clear()
        
        logger.info("SPC analyzer data reset")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics"""
        if len(self.data_buffer) == 0:
            return {'no_data': True}
        
        data = list(self.data_buffer)
        
        defect_counts = [d['defect_count'] for d in data]
        defect_rates = [d['defect_rate'] for d in data]
        confidences = [d['avg_confidence'] for d in data]
        
        return {
            'sample_count': len(data),
            'time_span_hours': (data[-1]['timestamp'] - data[0]['timestamp']).total_seconds() / 3600,
            'defect_count_stats': {
                'mean': np.mean(defect_counts),
                'median': np.median(defect_counts),
                'std': np.std(defect_counts),
                'min': np.min(defect_counts),
                'max': np.max(defect_counts)
            },
            'defect_rate_stats': {
                'mean': np.mean(defect_rates),
                'median': np.median(defect_rates),
                'std': np.std(defect_rates),
                'min': np.min(defect_rates),
                'max': np.max(defect_rates)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }
