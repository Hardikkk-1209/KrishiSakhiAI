"""
Synthetic Livestock IoT Sensor Data Generator
===============================================
Generates realistic IoT sensor data for livestock health monitoring.
Simulates normal, at-risk, and critical health conditions across
multiple disease profiles for training ML models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class LivestockDataGenerator:
    """Generates synthetic IoT sensor data for livestock health monitoring."""

    BREEDS = [
        'Holstein Friesian', 'Jersey', 'Sahiwal', 'Gir', 
        'Red Sindhi', 'Murrah Buffalo', 'Tharparkar', 'Crossbred'
    ]

    AGE_RANGE = (1.5, 12.0)  # years

    # Normal physiological ranges for cattle
    NORMAL_RANGES = {
        'body_temp':        (38.3, 39.4),    # °C
        'heart_rate':       (40, 80),         # bpm
        'respiratory_rate': (15, 30),         # breaths/min
        'activity_level':   (50, 85),         # score 0-100
        'rumination_min':   (380, 520),       # min/day
        'feed_intake':      (18, 28),         # kg/day
        'water_intake':     (40, 80),         # L/day
        'milk_yield':       (12, 30),         # L/day
        'lying_time':       (10, 14),         # hrs/day
        'steps_count':      (2000, 5000),     # /day
    }

    # Disease-specific deviations from normal ranges
    DISEASE_PROFILES = {
        'Mastitis': {
            'body_temp':        (39.5, 41.0),
            'heart_rate':       (75, 100),
            'milk_yield':       (3, 12),
            'activity_level':   (20, 50),
            'rumination_min':   (200, 350),
            'feed_intake':      (10, 18),
        },
        'Lameness': {
            'activity_level':   (10, 35),
            'steps_count':      (200, 1200),
            'lying_time':       (15, 20),
            'feed_intake':      (12, 20),
            'rumination_min':   (250, 380),
        },
        'BRD': {  # Bovine Respiratory Disease
            'body_temp':        (40.0, 41.5),
            'respiratory_rate': (35, 60),
            'heart_rate':       (80, 110),
            'feed_intake':      (5, 14),
            'activity_level':   (15, 40),
            'rumination_min':   (150, 300),
        },
        'Heat_Stress': {
            'body_temp':        (39.8, 41.0),
            'respiratory_rate': (40, 80),
            'water_intake':     (80, 120),
            'milk_yield':       (5, 15),
            'activity_level':   (15, 35),
            'rumination_min':   (150, 300),
        },
        'Metabolic_Disorder': {
            'feed_intake':      (3, 12),
            'milk_yield':       (4, 12),
            'rumination_min':   (100, 250),
            'activity_level':   (15, 40),
            'body_temp':        (38.0, 38.5),
            'heart_rate':       (85, 110),
        },
    }

    # Gait score mapping: 1=Normal, 5=Severe Lameness
    GAIT_PROFILES = {
        'Normal':           {'gait_score': (1.0, 1.5), 'stance_symmetry': (0.90, 1.0), 'stride_length': (1.4, 1.7)},
        'Mild_Lameness':    {'gait_score': (2.0, 2.5), 'stance_symmetry': (0.75, 0.90), 'stride_length': (1.1, 1.4)},
        'Moderate_Lameness':{'gait_score': (3.0, 3.5), 'stance_symmetry': (0.55, 0.75), 'stride_length': (0.7, 1.1)},
        'Severe_Lameness':  {'gait_score': (4.0, 5.0), 'stance_symmetry': (0.30, 0.55), 'stride_length': (0.3, 0.7)},
    }

    def __init__(self, seed=42):
        """Initialize the data generator with a random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def _sample_range(self, low, high, n=1, noise_factor=0.02):
        """Sample values uniformly within a range with optional Gaussian noise."""
        values = np.random.uniform(low, high, n)
        noise = np.random.normal(0, (high - low) * noise_factor, n)
        return np.clip(values + noise, low * 0.8, high * 1.2)

    def _generate_animal_profile(self, animal_id):
        """Generate a static profile for an animal."""
        return {
            'animal_id': f"KS-{animal_id:04d}",
            'breed': random.choice(self.BREEDS),
            'age_years': round(np.random.uniform(*self.AGE_RANGE), 1),
            'weight_kg': round(np.random.uniform(350, 650), 1),
            'lactation_number': np.random.randint(0, 7),
        }

    def _generate_healthy_reading(self, profile):
        """Generate a single healthy sensor reading for an animal."""
        reading = {}
        for feature, (low, high) in self.NORMAL_RANGES.items():
            reading[feature] = round(self._sample_range(low, high, 1)[0], 2)

        # Gait data for healthy animal
        gait = self.GAIT_PROFILES['Normal']
        reading['gait_score'] = round(self._sample_range(*gait['gait_score'], 1)[0], 1)
        reading['stance_symmetry'] = round(self._sample_range(*gait['stance_symmetry'], 1)[0], 3)
        reading['stride_length'] = round(self._sample_range(*gait['stride_length'], 1)[0], 2)

        # Health labels
        reading['health_status'] = 0  # Healthy
        reading['disease_type'] = 'None'
        reading['is_anomaly'] = 0

        return reading

    def _generate_diseased_reading(self, profile, disease, severity='moderate'):
        """Generate a sensor reading for a diseased animal."""
        # Start with normal values
        reading = self._generate_healthy_reading(profile)

        # Override with disease-specific values
        disease_profile = self.DISEASE_PROFILES.get(disease, {})
        for feature, (low, high) in disease_profile.items():
            if severity == 'mild':
                # Blend: 60% normal, 40% diseased
                normal_val = reading[feature]
                diseased_val = self._sample_range(low, high, 1)[0]
                reading[feature] = round(0.6 * normal_val + 0.4 * diseased_val, 2)
            elif severity == 'severe':
                reading[feature] = round(self._sample_range(low, high, 1, noise_factor=0.05)[0], 2)
            else:  # moderate
                normal_val = reading[feature]
                diseased_val = self._sample_range(low, high, 1)[0]
                reading[feature] = round(0.3 * normal_val + 0.7 * diseased_val, 2)

        # Gait data — affected more if lameness
        if disease == 'Lameness':
            if severity == 'mild':
                gait_profile = self.GAIT_PROFILES['Mild_Lameness']
            elif severity == 'severe':
                gait_profile = self.GAIT_PROFILES['Severe_Lameness']
            else:
                gait_profile = self.GAIT_PROFILES['Moderate_Lameness']
        elif disease in ['BRD', 'Heat_Stress']:
            gait_profile = self.GAIT_PROFILES['Mild_Lameness']
        else:
            gait_profile = self.GAIT_PROFILES['Normal']

        reading['gait_score'] = round(self._sample_range(*gait_profile['gait_score'], 1)[0], 1)
        reading['stance_symmetry'] = round(self._sample_range(*gait_profile['stance_symmetry'], 1)[0], 3)
        reading['stride_length'] = round(self._sample_range(*gait_profile['stride_length'], 1)[0], 2)

        # Health labels
        if severity == 'mild':
            reading['health_status'] = 1  # At-Risk
        else:
            reading['health_status'] = 2  # Critical

        reading['disease_type'] = disease
        reading['is_anomaly'] = 1

        return reading

    def generate_dataset(self, n_animals=200, n_days=30, anomaly_rate=0.20):
        """
        Generate a full synthetic dataset.

        Args:
            n_animals: Number of animals
            n_days: Number of days of data
            anomaly_rate: Fraction of readings that are anomalous (diseased)

        Returns:
            pd.DataFrame with all sensor readings, profiles, and labels
        """
        records = []
        diseases = list(self.DISEASE_PROFILES.keys())
        severities = ['mild', 'moderate', 'severe']

        base_date = datetime(2025, 1, 1)

        for i in range(n_animals):
            profile = self._generate_animal_profile(i)

            # Decide if this animal will have episodes of illness
            has_illness = np.random.random() < anomaly_rate * 3  # Some animals more prone
            illness_days = set()
            illness_disease = None
            illness_severity = None

            if has_illness:
                illness_disease = random.choice(diseases)
                illness_severity = random.choice(severities)
                # Illness lasts 3-10 consecutive days
                onset = np.random.randint(0, max(1, n_days - 10))
                duration = np.random.randint(3, min(11, n_days - onset + 1))
                illness_days = set(range(onset, onset + duration))

            for day in range(n_days):
                timestamp = base_date + timedelta(days=day, hours=np.random.randint(6, 20),
                                                   minutes=np.random.randint(0, 60))

                if day in illness_days and illness_disease:
                    # Progressive severity: mild at start, worsening
                    progress = (day - min(illness_days)) / max(1, len(illness_days) - 1)
                    if progress < 0.3:
                        sev = 'mild'
                    elif progress < 0.7:
                        sev = 'moderate'
                    else:
                        sev = illness_severity
                    reading = self._generate_diseased_reading(profile, illness_disease, sev)
                else:
                    reading = self._generate_healthy_reading(profile)

                # Add profile and timestamp
                record = {**profile, 'timestamp': timestamp, 'day': day, **reading}

                # Environmental context
                record['ambient_temp'] = round(np.random.uniform(22, 40), 1)
                record['humidity_pct'] = round(np.random.uniform(40, 90), 1)
                record['thi_index'] = round(
                    0.8 * record['ambient_temp'] + record['humidity_pct'] / 100 *
                    (record['ambient_temp'] - 14.4) + 46.4, 1
                )  # Temperature Humidity Index

                records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values(['animal_id', 'timestamp']).reset_index(drop=True)

        print(f"✅ Generated dataset: {len(df)} records")
        print(f"   Animals: {n_animals} | Days: {n_days}")
        print(f"   Healthy: {(df['health_status'] == 0).sum()} | "
              f"At-Risk: {(df['health_status'] == 1).sum()} | "
              f"Critical: {(df['health_status'] == 2).sum()}")
        print(f"   Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")

        return df

    def generate_realtime_reading(self, animal_id=None, health='healthy', disease=None, severity='moderate'):
        """
        Generate a single real-time sensor reading (for dashboard simulation).
        
        Args:
            animal_id: Optional specific animal ID
            health: 'healthy' or 'diseased'
            disease: Disease type if health='diseased'
            severity: 'mild', 'moderate', or 'severe'
        
        Returns:
            dict with sensor readings
        """
        aid = animal_id or f"KS-{np.random.randint(0, 200):04d}"
        profile = self._generate_animal_profile(int(aid.split('-')[1]))
        profile['animal_id'] = aid

        if health == 'diseased' and disease:
            reading = self._generate_diseased_reading(profile, disease, severity)
        else:
            reading = self._generate_healthy_reading(profile)

        reading['timestamp'] = datetime.now()
        reading['ambient_temp'] = round(np.random.uniform(22, 40), 1)
        reading['humidity_pct'] = round(np.random.uniform(40, 90), 1)
        reading['thi_index'] = round(
            0.8 * reading['ambient_temp'] + reading['humidity_pct'] / 100 *
            (reading['ambient_temp'] - 14.4) + 46.4, 1
        )

        return {**profile, **reading}


if __name__ == "__main__":
    gen = LivestockDataGenerator(seed=42)
    df = gen.generate_dataset(n_animals=200, n_days=30, anomaly_rate=0.20)
    df.to_csv("livestock_training_data.csv", index=False)
    print(f"\n📊 Saved to livestock_training_data.csv")
    print(f"\nDisease distribution:\n{df['disease_type'].value_counts()}")
    print(f"\nHealth status distribution:\n{df['health_status'].value_counts()}")
