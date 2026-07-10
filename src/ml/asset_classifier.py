"""
Machine Learning Asset Classifier
Train and use ML models for asset recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Model cache directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'asset_classifier.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')


class AssetClassifier:
    """
    Machine Learning based asset classifier.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.classes = []

        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)

    def prepare_features(self, assets_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for ML model.

        Args:
            assets_df: DataFrame with asset data

        Returns:
            Feature matrix
        """
        features = []

        for _, asset in assets_df.iterrows():
            # Extract features
            height = asset.get('height_m', 0)
            lat = asset.get('lat', 0)
            lon = asset.get('lon', 0)
            confidence = asset.get('confidence', 0.5)

            # Create feature vector
            feature_vector = [
                height,
                lat,
                lon,
                confidence,
                height * confidence,  # Interaction feature
                abs(lat - lon),  # Difference feature
                height ** 2,  # Quadratic feature
            ]

            features.append(feature_vector)

        return np.array(features)

    def train(self, assets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the classifier.

        Args:
            assets_df: DataFrame with asset data

        Returns:
            Training metrics
        """
        # Prepare features and labels
        X = self.prepare_features(assets_df)
        y = assets_df['type'].values

        # Encode labels
        self.classes = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(self.classes)}
        y_encoded = np.array([label_map[label] for label in y])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model
        self._save_model()

        return {
            'accuracy': accuracy,
            'classes': self.classes,
            'n_samples': len(assets_df),
            'n_features': X.shape[1],
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.classes
            )
        }

    def predict(self, assets_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict asset types.

        Args:
            assets_df: DataFrame with asset data

        Returns:
            Predictions and probabilities
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            print("⚠️ Model not trained. Training now...")
            self.train(assets_df)

        # Prepare features
        X = self.prepare_features(assets_df)
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Decode labels
        label_map_inv = {idx: label for idx, label in enumerate(self.classes)}
        pred_labels = np.array([label_map_inv[p] for p in predictions])

        return pred_labels, probabilities

    def _save_model(self):
        """Save model to disk."""
        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.classes, os.path.join(MODEL_DIR, 'classes.pkl'))
        print(f"💾 Model saved to {MODEL_FILE}")

    def _load_model(self):
        """Load model from disk."""
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                self.classes = joblib.load(os.path.join(MODEL_DIR, 'classes.pkl'))
                print(f"📦 Model loaded from {MODEL_FILE}")
                return True
            except Exception as e:
                print(f"⚠️ Model load failed: {str(e)}")
                return False
        return False

    def predict_with_confidence(self, assets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with confidence scores.

        Args:
            assets_df: DataFrame with asset data

        Returns:
            DataFrame with predictions and confidence
        """
        pred_labels, probs = self.predict(assets_df)

        # Get confidence (max probability)
        confidence = np.max(probs, axis=1)

        # Create result DataFrame
        results = assets_df[['id', 'lat', 'lon', 'height_m']].copy()
        results['ml_predicted_type'] = pred_labels
        results['ml_confidence'] = confidence
        results['is_ml_valid'] = confidence > 0.6

        return results


def train_asset_classifier(assets_df: pd.DataFrame) -> Dict[str, Any]:
    """Train the asset classifier on given data."""
    classifier = AssetClassifier()
    return classifier.train(assets_df)


def predict_assets(assets_df: pd.DataFrame) -> pd.DataFrame:
    """Predict asset types using ML model."""
    classifier = AssetClassifier()
    return classifier.predict_with_confidence(assets_df)