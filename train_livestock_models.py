"""
🐄 Livestock Biosecurity — Model Training Pipeline
=====================================================
Generates synthetic data, trains all ML models, evaluates performance,
and saves trained models to disk for use by the dashboard.

Usage:
    python train_livestock_models.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livestock_biosecurity.data_generator import LivestockDataGenerator
from livestock_biosecurity.models import (
    HealthPredictor,
    AnomalyDetector,
    GaitPredictor,
    DiseaseForecaster,
    save_models,
)


def main():
    print("=" * 65)
    print("🐄  LIVESTOCK BIOSECURITY — MODEL TRAINING PIPELINE")
    print("=" * 65)
    start_time = time.time()

    # ── Step 1: Generate Synthetic Data ──────────────────────────────
    print("\n📊 Step 1: Generating Synthetic IoT Sensor Data...")
    print("-" * 50)
    generator = LivestockDataGenerator(seed=42)
    df = generator.generate_dataset(n_animals=250, n_days=30, anomaly_rate=0.20)

    # Save training data
    data_path = "livestock_training_data.csv"
    df.to_csv(data_path, index=False)
    print(f"   📁 Training data saved → {data_path}")
    print(f"   📏 Shape: {df.shape}")

    # ── Step 2: Train Health Predictor ───────────────────────────────
    print("\n🏥 Step 2: Training Health Risk Classifier (Random Forest)...")
    print("-" * 50)
    health_model = HealthPredictor()
    health_metrics = health_model.train(df)
    print(f"\n   📋 Classification Report:\n{health_metrics['report']}")

    # Feature importance (top 10)
    importances = sorted(health_metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    print("   🔑 Top 10 Feature Importances:")
    for feat, imp in importances[:10]:
        bar = '█' * int(imp * 100)
        print(f"      {feat:30s} {imp:.4f} {bar}")

    # ── Step 3: Train Anomaly Detector ───────────────────────────────
    print("\n🔍 Step 3: Training Anomaly Detector (Isolation Forest)...")
    print("-" * 50)
    anomaly_model = AnomalyDetector(contamination=0.08)
    anomaly_metrics = anomaly_model.train(df)

    # ── Step 4: Train Gait Predictor ─────────────────────────────────
    print("\n🦿 Step 4: Training Gait Score Predictor (Gradient Boosting)...")
    print("-" * 50)
    gait_model = GaitPredictor()
    gait_metrics = gait_model.train(df)

    # ── Step 5: Train Disease Forecaster ─────────────────────────────
    print("\n🦠 Step 5: Training Disease Forecaster (Gradient Boosting)...")
    print("-" * 50)
    disease_model = DiseaseForecaster()
    disease_metrics = disease_model.train(df)
    print(f"   Disease classes: {disease_metrics['classes']}")

    # ── Step 6: Save All Models ──────────────────────────────────────
    print("\n💾 Step 6: Saving Trained Models...")
    print("-" * 50)
    models_dict = {
        'health_predictor': health_model,
        'anomaly_detector': anomaly_model,
        'gait_predictor': gait_model,
        'disease_forecaster': disease_model,
    }
    save_models(models_dict, save_dir='trained_models')

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("✅  TRAINING COMPLETE — MODEL PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"""
    ┌──────────────────────────────────────────────────────┐
    │  Model                    │ Metric        │ Score    │
    ├──────────────────────────────────────────────────────┤
    │  Health Predictor (RF)    │ Accuracy      │ {health_metrics['accuracy']:.4f}   │
    │                           │ F1 (weighted) │ {health_metrics['f1_weighted']:.4f}   │
    ├──────────────────────────────────────────────────────┤
    │  Anomaly Detector (IF)    │ Precision     │ {anomaly_metrics['precision']:.4f}   │
    │                           │ Recall        │ {anomaly_metrics['recall']:.4f}   │
    │                           │ F1            │ {anomaly_metrics['f1']:.4f}   │
    ├──────────────────────────────────────────────────────┤
    │  Gait Predictor (GBR)     │ MAE           │ {gait_metrics['mae']:.4f}   │
    │                           │ R²            │ {gait_metrics['r2']:.4f}   │
    ├──────────────────────────────────────────────────────┤
    │  Disease Forecaster (GBC) │ Accuracy      │ {disease_metrics['accuracy']:.4f}   │
    │                           │ F1 (weighted) │ {disease_metrics['f1_weighted']:.4f}   │
    └──────────────────────────────────────────────────────┘

    ⏱️  Total training time: {elapsed:.1f}s
    📁  Models saved to: trained_models/
    📊  Training data: {data_path} ({len(df)} records)
    """)

    print("🚀 Ready! Launch the dashboard with:")
    print("   streamlit run livestock_app.py")
    print()


if __name__ == "__main__":
    main()
