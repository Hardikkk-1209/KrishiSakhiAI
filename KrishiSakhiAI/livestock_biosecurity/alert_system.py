"""
Biosecurity Alert System
==========================
Generates, manages, and prioritizes biosecurity alerts based on
ML model predictions, CV analysis, and IoT sensor anomalies.
"""

from datetime import datetime
from collections import defaultdict
import uuid


class BiosecurityAlertSystem:
    """
    Centralized alert system that aggregates inputs from all ML models
    and generates prioritized biosecurity alerts.
    """

    SEVERITY_LEVELS = {
        'CRITICAL': {'priority': 1, 'color': '#ef4444', 'icon': '🚨', 'response_time': '< 1 hour'},
        'HIGH':     {'priority': 2, 'color': '#f97316', 'icon': '⚠️', 'response_time': '< 4 hours'},
        'MEDIUM':   {'priority': 3, 'color': '#eab308', 'icon': '🔔', 'response_time': '< 12 hours'},
        'LOW':      {'priority': 4, 'color': '#22c55e', 'icon': 'ℹ️', 'response_time': '< 24 hours'},
    }

    ALERT_CATEGORIES = {
        'HEALTH_RISK':      'Animal Health Risk Detected',
        'ANOMALY':          'Behavioral Anomaly Detected',
        'LAMENESS':         'Lameness / Gait Issue Detected',
        'DISEASE_RISK':     'Disease Outbreak Risk',
        'RESPIRATORY':      'Respiratory Distress Detected',
        'ISOLATION':        'Social Isolation Detected',
        'BIOSECURITY':      'Biosecurity Protocol Alert',
        'ENVIRONMENTAL':    'Environmental Stress Alert',
    }

    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.alert_counts = defaultdict(int)

    def generate_alert(self, animal_id, category, severity, message, details=None, model_source=None):
        """
        Generate a new biosecurity alert.
        
        Args:
            animal_id: ID of the affected animal
            category: Alert category (from ALERT_CATEGORIES)
            severity: Severity level (CRITICAL/HIGH/MEDIUM/LOW)
            message: Human-readable alert message
            details: Additional details dict
            model_source: Which ML model generated this alert
        
        Returns:
            dict with full alert information
        """
        alert = {
            'alert_id': str(uuid.uuid4())[:8].upper(),
            'timestamp': datetime.now().isoformat(),
            'animal_id': animal_id,
            'category': category,
            'category_label': self.ALERT_CATEGORIES.get(category, category),
            'severity': severity,
            'severity_info': self.SEVERITY_LEVELS.get(severity, self.SEVERITY_LEVELS['LOW']),
            'message': message,
            'details': details or {},
            'model_source': model_source,
            'status': 'ACTIVE',
            'acknowledged': False,
        }

        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.alert_counts[severity] += 1

        return alert

    def process_health_prediction(self, animal_id, health_result):
        """Generate alerts from health prediction results."""
        alerts = []
        status = health_result.get('status', 0)
        label = health_result.get('status_label', 'Unknown')
        confidence = health_result.get('confidence', 0)

        if status == 2:  # Critical
            alert = self.generate_alert(
                animal_id=animal_id,
                category='HEALTH_RISK',
                severity='CRITICAL',
                message=f"Critical health status detected for {animal_id}. "
                        f"Confidence: {confidence*100:.0f}%. Immediate veterinary attention required.",
                details=health_result,
                model_source='HealthPredictor'
            )
            alerts.append(alert)
        elif status == 1:  # At-Risk
            alert = self.generate_alert(
                animal_id=animal_id,
                category='HEALTH_RISK',
                severity='HIGH',
                message=f"At-risk health status for {animal_id}. "
                        f"Confidence: {confidence*100:.0f}%. Monitor closely and schedule check-up.",
                details=health_result,
                model_source='HealthPredictor'
            )
            alerts.append(alert)

        return alerts

    def process_anomaly_detection(self, animal_id, anomaly_result):
        """Generate alerts from anomaly detection results."""
        alerts = []

        if anomaly_result.get('is_anomaly'):
            severity_map = {'High': 'CRITICAL', 'Medium': 'HIGH', 'Low': 'MEDIUM'}
            anomaly_severity = anomaly_result.get('severity', 'Low')

            alert = self.generate_alert(
                animal_id=animal_id,
                category='ANOMALY',
                severity=severity_map.get(anomaly_severity, 'MEDIUM'),
                message=f"Anomalous behavior pattern detected for {animal_id}. "
                        f"Anomaly score: {anomaly_result.get('anomaly_score', 0):.3f}. "
                        f"Severity: {anomaly_severity}.",
                details=anomaly_result,
                model_source='AnomalyDetector'
            )
            alerts.append(alert)

        return alerts

    def process_gait_analysis(self, animal_id, gait_result):
        """Generate alerts from gait analysis results."""
        alerts = []

        if gait_result.get('needs_attention') or gait_result.get('needs_intervention'):
            score = gait_result.get('gait_score', gait_result.get('locomotion_score', 1))
            label = gait_result.get('lameness_label', gait_result.get('label', 'Unknown'))

            if score >= 4:
                severity = 'CRITICAL'
            elif score >= 3:
                severity = 'HIGH'
            else:
                severity = 'MEDIUM'

            alert = self.generate_alert(
                animal_id=animal_id,
                category='LAMENESS',
                severity=severity,
                message=f"Lameness detected for {animal_id}. "
                        f"Gait score: {score:.1f}/5.0 — {label}. "
                        f"Recommend hoof examination.",
                details=gait_result,
                model_source='GaitAnalyzer'
            )
            alerts.append(alert)

        return alerts

    def process_disease_forecast(self, animal_id, disease_result):
        """Generate alerts from disease forecasting results."""
        alerts = []

        if not disease_result.get('is_healthy', True):
            disease = disease_result.get('predicted_disease', 'Unknown')
            confidence = disease_result.get('confidence', 0)

            severity = 'CRITICAL' if confidence > 0.75 else ('HIGH' if confidence > 0.5 else 'MEDIUM')

            # Disease-specific messages
            disease_actions = {
                'Mastitis': 'Isolate animal, check udder, initiate treatment protocol.',
                'Lameness': 'Schedule hoof trimming, reduce walking distance, provide soft bedding.',
                'BRD': 'Isolate immediately, monitor respiratory rate, consult veterinarian.',
                'Heat_Stress': 'Move to shaded area, increase water access, activate cooling systems.',
                'Metabolic_Disorder': 'Adjust feed ration, administer supplements, monitor closely.',
            }

            action = disease_actions.get(disease, 'Monitor closely and consult veterinarian.')

            alert = self.generate_alert(
                animal_id=animal_id,
                category='DISEASE_RISK',
                severity=severity,
                message=f"Disease risk: {disease.replace('_', ' ')} predicted for {animal_id}. "
                        f"Confidence: {confidence*100:.0f}%. Action: {action}",
                details=disease_result,
                model_source='DiseaseForecaster'
            )
            alerts.append(alert)

        return alerts

    def process_behavior_analysis(self, animal_id, behavior_result):
        """Generate alerts from behavior analysis results."""
        alerts = []

        if behavior_result.get('is_abnormal'):
            pattern = behavior_result.get('behavior_pattern', 'Unknown')
            health_score = behavior_result.get('behavior_health_score', 100)

            if health_score < 30:
                severity = 'CRITICAL'
            elif health_score < 50:
                severity = 'HIGH'
            else:
                severity = 'MEDIUM'

            category_map = {
                'Respiratory_Distress': 'RESPIRATORY',
                'Isolation': 'ISOLATION',
                'Lethargy': 'ANOMALY',
            }

            alert = self.generate_alert(
                animal_id=animal_id,
                category=category_map.get(pattern, 'ANOMALY'),
                severity=severity,
                message=f"Behavioral change detected for {animal_id}: "
                        f"{pattern.replace('_', ' ')}. Health score: {health_score:.0f}/100.",
                details=behavior_result,
                model_source='BehaviorAnalyzer'
            )
            alerts.append(alert)

        # Process sub-alerts from CV analysis
        for sub_alert in behavior_result.get('alerts', []):
            alert = self.generate_alert(
                animal_id=animal_id,
                category='BIOSECURITY',
                severity=sub_alert.get('severity', 'MEDIUM'),
                message=sub_alert.get('message', ''),
                details=sub_alert,
                model_source='BehaviorAnalyzer'
            )
            alerts.append(alert)

        return alerts

    def get_active_alerts(self, severity=None, category=None, limit=50):
        """Get active alerts, optionally filtered."""
        alerts = [a for a in self.active_alerts if a['status'] == 'ACTIVE']

        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        if category:
            alerts = [a for a in alerts if a['category'] == category]

        # Sort by priority (severity)
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        alerts.sort(key=lambda a: priority_order.get(a['severity'], 4))

        return alerts[:limit]

    def get_alert_summary(self):
        """Get a summary of current alert status."""
        active = [a for a in self.active_alerts if a['status'] == 'ACTIVE']

        return {
            'total_active': len(active),
            'by_severity': {
                'CRITICAL': len([a for a in active if a['severity'] == 'CRITICAL']),
                'HIGH': len([a for a in active if a['severity'] == 'HIGH']),
                'MEDIUM': len([a for a in active if a['severity'] == 'MEDIUM']),
                'LOW': len([a for a in active if a['severity'] == 'LOW']),
            },
            'by_category': {
                cat: len([a for a in active if a['category'] == cat])
                for cat in set(a['category'] for a in active)
            },
            'total_historical': len(self.alert_history),
            'unique_animals_affected': len(set(a['animal_id'] for a in active)),
        }

    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False

    def resolve_alert(self, alert_id, resolution_note=''):
        """Resolve and close an alert."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'RESOLVED'
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolution_note'] = resolution_note
                return True
        return False

    def clear_all_alerts(self):
        """Clear all active alerts (for demo / reset purposes)."""
        self.active_alerts = []
        self.alert_counts = defaultdict(int)
