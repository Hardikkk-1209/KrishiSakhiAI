"""
Computer Vision Module for Livestock Monitoring
==================================================
Simulates computer vision-based analysis for:
  - Gait analysis and lameness detection
  - Behavioral change detection (respiratory distress, lethargy)
  - Posture and mobility scoring
  - Herd-level anomaly flagging

In a production system, this would process video feeds from barn cameras.
Here we simulate the CV pipeline outputs using rule-based + noise models.
"""

import numpy as np
from datetime import datetime
import random


class GaitAnalyzer:
    """
    Simulates computer vision gait analysis from video feeds.
    
    In production, this would use pose estimation (e.g., DeepLabCut)
    on video frames to extract keypoints and compute gait metrics.
    """

    # Locomotion Scoring System (1-5 scale, industry standard)
    LOCOMOTION_SCORES = {
        1: {'label': 'Normal',            'description': 'Walks freely, flat back, makes long confident strides'},
        2: {'label': 'Mildly Lame',       'description': 'Slight asymmetry, shortened stride, slight head bob'},
        3: {'label': 'Moderately Lame',   'description': 'Obvious asymmetry, arched back, reluctance to bear weight'},
        4: {'label': 'Lame',              'description': 'Cannot walk properly, strong reluctance, visible discomfort'},
        5: {'label': 'Severely Lame',     'description': 'Cannot walk or bear weight, refuses to move'},
    }

    def __init__(self):
        self.analysis_history = []

    def analyze_gait(self, sensor_data):
        """
        Analyze gait from sensor data (simulating CV output).
        
        In production: input would be video frames → pose estimation → gait metrics.
        Here: we compute from IoT sensors as a simulation.
        
        Args:
            sensor_data: dict with activity_level, steps_count, lying_time, etc.
        
        Returns:
            dict with gait analysis results
        """
        activity = sensor_data.get('activity_level', 70)
        steps = sensor_data.get('steps_count', 3500)
        lying_time = sensor_data.get('lying_time', 12)
        stride_length = sensor_data.get('stride_length', 1.5)
        stance_symmetry = sensor_data.get('stance_symmetry', 0.95)

        # Compute composite gait metrics (simulating CV output)
        # Normalized scores (0-1, lower = worse)
        activity_score = np.clip(activity / 85, 0, 1)
        step_score = np.clip(steps / 4000, 0, 1)
        rest_score = np.clip(1 - (lying_time - 12) / 8, 0, 1)  # Penalize excessive rest
        stride_score = np.clip(stride_length / 1.6, 0, 1)
        symmetry_score = np.clip(stance_symmetry, 0, 1)

        # Weighted composite → locomotion score
        composite = (
            0.15 * activity_score +
            0.15 * step_score +
            0.15 * rest_score +
            0.25 * stride_score +
            0.30 * symmetry_score
        )

        # Add slight noise (simulating CV measurement variance)
        noise = np.random.normal(0, 0.03)
        composite = np.clip(composite + noise, 0, 1)

        # Map composite to locomotion score (1-5)
        if composite > 0.85:
            locomotion_score = 1
        elif composite > 0.70:
            locomotion_score = 2
        elif composite > 0.50:
            locomotion_score = 3
        elif composite > 0.30:
            locomotion_score = 4
        else:
            locomotion_score = 5

        score_info = self.LOCOMOTION_SCORES[locomotion_score]

        # Detailed CV metrics (simulated)
        cv_metrics = {
            'back_posture_angle': round(180 - (locomotion_score - 1) * 8 + np.random.normal(0, 2), 1),
            'head_bob_amplitude': round(max(0, (locomotion_score - 1) * 3.5 + np.random.normal(0, 1)), 1),
            'stride_regularity': round(composite * 100, 1),
            'weight_distribution': round(symmetry_score * 100, 1),
            'walking_speed_mps': round(max(0.2, 1.4 - (locomotion_score - 1) * 0.25 + np.random.normal(0, 0.1)), 2),
        }

        result = {
            'locomotion_score': locomotion_score,
            'label': score_info['label'],
            'description': score_info['description'],
            'composite_score': round(composite * 100, 1),
            'component_scores': {
                'activity': round(activity_score * 100, 1),
                'steps': round(step_score * 100, 1),
                'rest_pattern': round(rest_score * 100, 1),
                'stride': round(stride_score * 100, 1),
                'symmetry': round(symmetry_score * 100, 1),
            },
            'cv_metrics': cv_metrics,
            'needs_intervention': locomotion_score >= 3,
            'urgency': 'Critical' if locomotion_score >= 4 else ('Warning' if locomotion_score >= 3 else 'Normal'),
            'timestamp': datetime.now().isoformat(),
        }

        self.analysis_history.append(result)
        return result

    def get_herd_gait_summary(self, herd_readings):
        """Analyze gait for an entire herd."""
        results = [self.analyze_gait(r) for r in herd_readings]

        score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in results:
            score_counts[r['locomotion_score']] += 1

        total = len(results)
        return {
            'total_animals': total,
            'score_distribution': score_counts,
            'normal_pct': round((score_counts[1] / total) * 100, 1) if total > 0 else 0,
            'lame_pct': round(((score_counts[3] + score_counts[4] + score_counts[5]) / total) * 100, 1) if total > 0 else 0,
            'critical_count': score_counts[4] + score_counts[5],
            'avg_locomotion_score': round(np.mean([r['locomotion_score'] for r in results]), 2),
            'individual_results': results,
        }


class BehaviorAnalyzer:
    """
    Simulates computer vision behavioral analysis.
    
    Detects behavioral changes that may indicate illness:
    - Respiratory distress (rapid flank movements)
    - Isolation from herd
    - Reduced feed/water access
    - Abnormal postures
    """

    BEHAVIOR_PATTERNS = {
        'Normal': {
            'social_score': (70, 100),
            'feeding_frequency': (8, 14),
            'posture_score': (85, 100),
            'respiratory_pattern': (0, 15),  # distress indicator (0=calm, 100=severe)
        },
        'Respiratory_Distress': {
            'social_score': (30, 60),
            'feeding_frequency': (3, 7),
            'posture_score': (40, 65),
            'respiratory_pattern': (50, 90),
        },
        'Lethargy': {
            'social_score': (20, 50),
            'feeding_frequency': (2, 6),
            'posture_score': (50, 70),
            'respiratory_pattern': (5, 25),
        },
        'Isolation': {
            'social_score': (5, 30),
            'feeding_frequency': (3, 8),
            'posture_score': (55, 75),
            'respiratory_pattern': (10, 35),
        },
    }

    def __init__(self):
        self.behavior_history = []

    def analyze_behavior(self, sensor_data):
        """
        Analyze animal behavior from sensor data (simulating CV output).
        
        Args:
            sensor_data: dict with IoT sensor readings
        
        Returns:
            dict with behavioral analysis results
        """
        activity = sensor_data.get('activity_level', 70)
        respiratory_rate = sensor_data.get('respiratory_rate', 22)
        feed_intake = sensor_data.get('feed_intake', 22)
        rumination = sensor_data.get('rumination_min', 450)
        body_temp = sensor_data.get('body_temp', 38.8)

        # Derive behavioral metrics from IoT data
        social_score = np.clip(activity * 1.2 + np.random.normal(0, 5), 0, 100)
        feeding_freq = np.clip(feed_intake * 0.5 + np.random.normal(0, 1), 1, 20)
        posture_score = np.clip(
            100 - abs(body_temp - 38.85) * 15 - max(0, respiratory_rate - 25) * 1.5 + np.random.normal(0, 3),
            0, 100
        )
        resp_distress = np.clip(
            max(0, respiratory_rate - 25) * 3 + max(0, body_temp - 39.5) * 20 + np.random.normal(0, 5),
            0, 100
        )

        # Determine behavior pattern
        if resp_distress > 50:
            pattern = 'Respiratory_Distress'
        elif social_score < 30:
            pattern = 'Isolation'
        elif activity < 30 and rumination < 300:
            pattern = 'Lethargy'
        else:
            pattern = 'Normal'

        # Compute overall behavioral health score (0-100)
        behavior_health = np.clip(
            0.25 * social_score +
            0.20 * (feeding_freq / 14 * 100) +
            0.25 * posture_score +
            0.30 * (100 - resp_distress),
            0, 100
        )

        result = {
            'behavior_pattern': pattern,
            'behavior_health_score': round(float(behavior_health), 1),
            'metrics': {
                'social_score': round(float(social_score), 1),
                'feeding_frequency': round(float(feeding_freq), 1),
                'posture_score': round(float(posture_score), 1),
                'respiratory_distress': round(float(resp_distress), 1),
            },
            'alerts': [],
            'is_abnormal': pattern != 'Normal',
            'timestamp': datetime.now().isoformat(),
        }

        # Generate specific alerts
        if resp_distress > 60:
            result['alerts'].append({
                'type': 'Respiratory Distress',
                'severity': 'HIGH',
                'message': f'Elevated respiratory distress indicator: {resp_distress:.0f}/100'
            })
        if social_score < 25:
            result['alerts'].append({
                'type': 'Social Isolation',
                'severity': 'MEDIUM',
                'message': f'Animal showing isolation behavior. Social score: {social_score:.0f}/100'
            })
        if feeding_freq < 5:
            result['alerts'].append({
                'type': 'Reduced Feeding',
                'severity': 'MEDIUM',
                'message': f'Feeding frequency significantly reduced: {feeding_freq:.0f}/day'
            })
        if posture_score < 50:
            result['alerts'].append({
                'type': 'Abnormal Posture',
                'severity': 'HIGH',
                'message': f'Abnormal posture detected. Score: {posture_score:.0f}/100'
            })

        self.behavior_history.append(result)
        return result

    def analyze_herd_behavior(self, herd_readings):
        """Analyze behavior patterns across the herd."""
        results = [self.analyze_behavior(r) for r in herd_readings]

        pattern_counts = {}
        for r in results:
            p = r['behavior_pattern']
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        total = len(results)
        all_alerts = []
        for r in results:
            all_alerts.extend(r['alerts'])

        return {
            'total_animals': total,
            'pattern_distribution': pattern_counts,
            'normal_pct': round((pattern_counts.get('Normal', 0) / total) * 100, 1) if total > 0 else 0,
            'abnormal_count': total - pattern_counts.get('Normal', 0),
            'avg_behavior_health': round(np.mean([r['behavior_health_score'] for r in results]), 1),
            'total_alerts': len(all_alerts),
            'high_severity_alerts': len([a for a in all_alerts if a['severity'] == 'HIGH']),
            'individual_results': results,
        }
