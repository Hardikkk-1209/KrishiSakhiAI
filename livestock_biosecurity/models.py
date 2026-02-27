"""
Livestock Health ML Models
============================
Trained ML models for livestock health monitoring:
  - HealthPredictor:    Random Forest classifier (Healthy / At-Risk / Critical)
  - AnomalyDetector:    Isolation Forest for unusual pattern detection
  - GaitPredictor:      Gradient Boosting for lameness scoring
  - DiseaseForecaster:  Multi-class classifier for disease type prediction
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# ── Feature columns used by all models ──────────────────────────────────────
SENSOR_FEATURES = [
    'body_temp', 'heart_rate', 'respiratory_rate', 'activity_level',
    'rumination_min', 'feed_intake', 'water_intake', 'milk_yield',
    'lying_time', 'steps_count',
]

GAIT_FEATURES = [
    'activity_level', 'steps_count', 'lying_time',
    'stride_length', 'stance_symmetry',
]

ALL_FEATURES = SENSOR_FEATURES + ['gait_score', 'stance_symmetry', 'stride_length',
                                    'ambient_temp', 'humidity_pct', 'thi_index']

HEALTH_LABELS = {0: 'Healthy', 1: 'At-Risk', 2: 'Critical'}


class HealthPredictor:
    """
    Random Forest classifier for livestock health status prediction.
    Classes: 0=Healthy, 1=At-Risk, 2=Critical
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.features = SENSOR_FEATURES + ['gait_score', 'stance_symmetry',
                                            'stride_length', 'ambient_temp',
                                            'humidity_pct', 'thi_index']
        self.is_trained = False
        self.metrics = {}

    def _engineer_features(self, df):
        """Create additional engineered features."""
        X = df[self.features].copy()

        # Engineered features
        X['temp_deviation'] = X['body_temp'] - 38.85  # deviation from normal mean
        X['activity_rumination_ratio'] = X['activity_level'] / (X['rumination_min'] + 1)
        X['feed_water_ratio'] = X['feed_intake'] / (X['water_intake'] + 1)
        X['cardio_stress'] = X['heart_rate'] * X['respiratory_rate'] / 1000
        X['mobility_score'] = (X['activity_level'] * X['steps_count']) / 10000
        X['rest_activity_ratio'] = X['lying_time'] / (X['activity_level'] + 1)

        return X

    def train(self, df):
        """Train the health prediction model."""
        X = self._engineer_features(df)
        y = df['health_status'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'report': classification_report(y_test, y_pred, target_names=['Healthy', 'At-Risk', 'Critical']),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
        }

        self.is_trained = True
        print(f"✅ HealthPredictor trained — Accuracy: {self.metrics['accuracy']:.3f} | "
              f"F1: {self.metrics['f1_weighted']:.3f}")
        return self.metrics

    def predict(self, reading_dict):
        """Predict health status from a single reading dict."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        df = pd.DataFrame([reading_dict])
        X = self._engineer_features(df)
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        return {
            'status': int(prediction),
            'status_label': HEALTH_LABELS[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                HEALTH_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
        }

    def predict_batch(self, df):
        """Predict health status for a DataFrame."""
        X = self._engineer_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)

        results = df.copy()
        results['predicted_status'] = predictions
        results['predicted_label'] = [HEALTH_LABELS[p] for p in predictions]
        results['confidence'] = probas.max(axis=1)
        return results


class AnomalyDetector:
    """
    Isolation Forest for detecting anomalous livestock behavior.
    Trained only on healthy data to flag deviations.
    """

    def __init__(self, contamination=0.05):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.features = SENSOR_FEATURES + ['gait_score', 'stance_symmetry', 'stride_length']
        self.is_trained = False

    def train(self, df):
        """Train on healthy data only."""
        healthy_data = df[df['health_status'] == 0]
        X = healthy_data[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True

        # Evaluate on full dataset
        X_all = self.scaler.transform(df[self.features].values)
        scores = self.model.decision_function(X_all)
        predictions = self.model.predict(X_all)

        # -1 = anomaly, 1 = normal
        true_anomalies = df['is_anomaly'].values
        detected = (predictions == -1).astype(int)

        tp = ((detected == 1) & (true_anomalies == 1)).sum()
        fp = ((detected == 1) & (true_anomalies == 0)).sum()
        fn = ((detected == 0) & (true_anomalies == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            'anomalies_detected': int(detected.sum()),
            'total_records': len(df),
        }

        print(f"✅ AnomalyDetector trained — Precision: {precision:.3f} | "
              f"Recall: {recall:.3f} | F1: {self.metrics['f1']:.3f}")
        return self.metrics

    def predict(self, reading_dict):
        """Check if a reading is anomalous."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        X = np.array([[reading_dict.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        score = self.model.decision_function(X_scaled)[0]

        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': round(float(-score), 4),  # Higher = more anomalous
            'severity': 'High' if score < -0.3 else ('Medium' if score < -0.1 else 'Low'),
        }


class GaitPredictor:
    """
    Gradient Boosting regressor for predicting gait (lameness) score.
    Score: 1.0 (Normal) → 5.0 (Severe Lameness)
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.features = GAIT_FEATURES
        self.is_trained = False

    def train(self, df):
        """Train gait score predictor."""
        X = df[self.features].values
        y = df['gait_score'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': self.model.score(X_test_scaled, y_test),
        }

        self.is_trained = True
        print(f"✅ GaitPredictor trained — MAE: {self.metrics['mae']:.3f} | "
              f"R²: {self.metrics['r2']:.3f}")
        return self.metrics

    def predict(self, reading_dict):
        """Predict gait score from a single reading."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        X = np.array([[reading_dict.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X)
        score = float(np.clip(self.model.predict(X_scaled)[0], 1.0, 5.0))

        if score < 2.0:
            label = 'Normal'
        elif score < 3.0:
            label = 'Mild Lameness'
        elif score < 4.0:
            label = 'Moderate Lameness'
        else:
            label = 'Severe Lameness'

        return {
            'gait_score': round(score, 2),
            'lameness_label': label,
            'needs_attention': score >= 2.5,
        }


class DiseaseForecaster:
    """
    Multi-class classifier for predicting specific disease types.
    Uses gradient boosting on sensor + gait features.
    """

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = ALL_FEATURES
        self.is_trained = False

    def train(self, df):
        """Train disease type predictor on diseased records only."""
        # Include all records — 'None' means healthy
        X = df[self.features].values
        y = self.label_encoder.fit_transform(df['disease_type'].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classes': list(self.label_encoder.classes_),
        }

        self.is_trained = True
        print(f"✅ DiseaseForecaster trained — Accuracy: {self.metrics['accuracy']:.3f} | "
              f"F1: {self.metrics['f1_weighted']:.3f}")
        return self.metrics

    def predict(self, reading_dict):
        """Predict potential disease type."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        X = np.array([[reading_dict.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        disease_label = self.label_encoder.inverse_transform([prediction])[0]
        prob_dict = {
            self.label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        return {
            'predicted_disease': disease_label,
            'confidence': float(max(probabilities)),
            'disease_probabilities': prob_dict,
            'is_healthy': disease_label == 'None',
        }


# ── Save / Load Utilities ───────────────────────────────────────────────────

def save_models(models_dict, save_dir='trained_models'):
    """Save all trained models to disk."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    for name, model in models_dict.items():
        filepath = save_path / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"   💾 Saved {name} → {filepath}")

    print(f"\n✅ All models saved to {save_path}/")


def load_models(save_dir='trained_models'):
    """Load all trained models from disk."""
    save_path = Path(save_dir)
    models = {}

    model_files = {
        'health_predictor': HealthPredictor,
        'anomaly_detector': AnomalyDetector,
        'gait_predictor': GaitPredictor,
        'disease_forecaster': DiseaseForecaster,
    }

    for name, cls in model_files.items():
        filepath = save_path / f"{name}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"   📂 Loaded {name} ← {filepath}")
        else:
            print(f"   ⚠️  {name} not found at {filepath}")

    return models
