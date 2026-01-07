"""
online_model.py - Online Learning Model for Purchase Intent Prediction

SGDClassifier kullanarak streaming data ile surekli model guncelleme.
Incremental learning (partial_fit) destekler.

Features:
    - Online learning with mini-batch updates
    - Model persistence (save/load)
    - Accuracy tracking over time
    - Thread-safe operations

Usage:
    from src.streaming.online_model import OnlinePredictor

    predictor = OnlinePredictor()
    prob = predictor.predict_proba({"view_count": 5, "cart_count": 1, ...})
    predictor.partial_fit(features, label=1)

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import threading
import logging

from .config import MODEL_CONFIG, FEATURE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlinePredictor:
    """
    Online Learning ile Purchase Intent Prediction.

    SGDClassifier (Stochastic Gradient Descent) kullanarak
    streaming data ile surekli model guncelleme yapar.

    Attributes:
        model: SGDClassifier instance
        scaler: StandardScaler for feature normalization
        is_fitted: Model egitildi mi?
        buffer_X: Mini-batch buffer for features
        buffer_y: Mini-batch buffer for labels
    """

    def __init__(self):
        """OnlinePredictor'i initialize et."""
        # Model: Logistic Regression with SGD
        self.model = SGDClassifier(
            loss='log_loss',            # Logistic regression for probabilities
            learning_rate='adaptive',   # Adapts learning rate
            eta0=MODEL_CONFIG["learning_rate"],
            random_state=42,
            warm_start=True,            # Enables incremental learning
            max_iter=1000,
            tol=1e-3
        )

        # Feature scaler
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Mini-batch buffer
        self.buffer_X = []
        self.buffer_y = []
        self.update_interval = MODEL_CONFIG["update_interval"]

        # Metrics tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.recent_accuracy = deque(maxlen=1000)  # Son 1000 tahmin

        # Thread safety
        self.lock = threading.Lock()

        # Load pre-trained model if exists
        self._load_model()

    def _load_model(self):
        """Kayitli model varsa yukle."""
        model_path = MODEL_CONFIG["model_path"]
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                    self.model = saved['model']
                    self.scaler = saved['scaler']
                    self.is_fitted = saved.get('is_fitted', True)
                    self.predictions_made = saved.get('predictions_made', 0)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        """Model'i kaydet."""
        model_path = MODEL_CONFIG["model_path"]
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_fitted': self.is_fitted,
                    'predictions_made': self.predictions_made
                }, f)
            logger.debug(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Could not save model: {e}")

    def _features_to_array(self, features):
        """
        Feature dict'i numpy array'e cevir.

        Args:
            features: dict with feature values

        Returns:
            np.ndarray: Shape (1, n_features)
        """
        return np.array([[
            features.get("view_count", 0),
            features.get("cart_count", 0),
            features.get("session_duration", 0),
            features.get("avg_price", 0),
            features.get("max_price", 0),
            features.get("unique_items", 0)
        ]], dtype=np.float64)

    def predict_proba(self, features):
        """
        Purchase olasiligi tahmin et.

        Args:
            features: dict with feature values

        Returns:
            float: Purchase probability (0-1)
        """
        X = self._features_to_array(features)

        with self.lock:
            if not self.is_fitted:
                # Model henuz fit edilmemis
                # Heuristic: cart_count > 0 ise yuksek olasilik
                cart_count = features.get("cart_count", 0)
                if cart_count > 0:
                    return 0.7  # Cart eklemis, yuksek olasilik
                elif features.get("view_count", 0) > 3:
                    return 0.3  # Cok bakti ama cart yok
                else:
                    return 0.1  # Az ilgi

            try:
                X_scaled = self.scaler.transform(X)
                proba = self.model.predict_proba(X_scaled)[0]

                # Class 1 (purchase) olasiligi
                # SGDClassifier.predict_proba returns array of shape (n_samples, n_classes)
                purchase_prob = proba[1] if len(proba) > 1 else proba[0]

                self.predictions_made += 1
                return float(purchase_prob)

            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                return 0.5  # Default probability

    def partial_fit(self, features, label):
        """
        Model'i yeni ornek ile guncelle (online learning).

        Mini-batch strategy: Buffer dolunca toplu update.

        Args:
            features: dict with feature values
            label: 0 or 1 (purchased or not)
        """
        X = self._features_to_array(features)

        with self.lock:
            # Buffer'a ekle
            self.buffer_X.append(X[0])
            self.buffer_y.append(label)

            # Accuracy tracking
            if self.is_fitted:
                pred = 1 if self.predict_proba(features) > 0.5 else 0
                is_correct = 1 if pred == label else 0
                self.recent_accuracy.append(is_correct)
                if is_correct:
                    self.correct_predictions += 1

            # Mini-batch update
            if len(self.buffer_X) >= self.update_interval:
                self._update_model()

    def _update_model(self):
        """Model'i buffer'daki verilerle guncelle."""
        if len(self.buffer_X) == 0:
            return

        X = np.array(self.buffer_X)
        y = np.array(self.buffer_y)

        try:
            if not self.is_fitted:
                # Ilk fit - scaler'i da fit et
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.model.fit(X_scaled, y)
                self.is_fitted = True
                logger.info(f"Initial model fit complete with {len(y)} samples")
            else:
                # Incremental update
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y, classes=[0, 1])
                logger.debug(f"Model updated with {len(y)} samples")

            # Buffer'i temizle
            self.buffer_X = []
            self.buffer_y = []

            # Periyodik kayit
            if self.predictions_made % 1000 == 0:
                self._save_model()

        except Exception as e:
            logger.error(f"Model update error: {e}")

    def force_update(self):
        """Buffer'daki tum verileri kullanarak model'i guncelle."""
        with self.lock:
            if len(self.buffer_X) > 0:
                self._update_model()
                self._save_model()

    def get_metrics(self):
        """
        Model performans metriklerini don.

        Returns:
            dict: Model metrics
        """
        with self.lock:
            if len(self.recent_accuracy) > 0:
                accuracy = sum(self.recent_accuracy) / len(self.recent_accuracy)
            else:
                accuracy = 0.0

            return {
                "predictions_made": self.predictions_made,
                "is_fitted": self.is_fitted,
                "buffer_size": len(self.buffer_X),
                "recent_accuracy": accuracy,
                "correct_predictions": self.correct_predictions,
                "model_type": "SGDClassifier (Online)"
            }

    def get_feature_importance(self):
        """
        Feature importance'lari don (eger model fitted ise).

        Returns:
            dict: Feature name -> importance
        """
        if not self.is_fitted:
            return {}

        try:
            # SGDClassifier icin coefficients
            coefs = self.model.coef_[0]
            importance = {}
            for i, name in enumerate(FEATURE_COLS):
                if i < len(coefs):
                    importance[name] = abs(coefs[i])
            return importance
        except Exception:
            return {}


# =============================================================================
# SIMPLE HEURISTIC PREDICTOR (Fallback)
# =============================================================================

class HeuristicPredictor:
    """
    Basit kural-tabanli predictor.
    Model egitilmeden once veya test icin kullanilir.
    """

    def __init__(self):
        self.predictions_made = 0

    def predict_proba(self, features):
        """
        Basit kurallarla purchase olasiligi tahmin et.
        """
        cart_count = features.get("cart_count", 0)
        view_count = features.get("view_count", 0)
        avg_price = features.get("avg_price", 0)

        self.predictions_made += 1

        # Kurallar
        if cart_count >= 3:
            return 0.85
        elif cart_count >= 1:
            return 0.60
        elif view_count >= 10:
            return 0.35
        elif view_count >= 5:
            return 0.20
        else:
            return 0.05

    def partial_fit(self, features, label):
        """Heuristic model guncellenmez."""
        pass

    def get_metrics(self):
        return {
            "predictions_made": self.predictions_made,
            "is_fitted": True,
            "buffer_size": 0,
            "recent_accuracy": 0.0,
            "model_type": "Heuristic (Rule-based)"
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_predictor(model_type=None):
    """
    Predictor instance olustur.

    Args:
        model_type: "sgd", "heuristic", or None (config'den)

    Returns:
        Predictor instance
    """
    model_type = model_type or MODEL_CONFIG.get("model_type", "sgd")

    if model_type == "heuristic":
        return HeuristicPredictor()
    else:
        return OnlinePredictor()
