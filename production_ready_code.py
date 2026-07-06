import logging
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


class BreastCancerANN:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def load_dataset(self):
        try:
            logging.info("Loading dataset...")
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            df.to_csv("breast_cancer.csv", index=False)
            logging.info("CSV saved as breast_cancer.csv")
            return df
        except Exception as e:
            raise Exception(f"Dataset Loading Error : {e}")

    def preprocess(self, df):
        try:
            X = df.drop("target", axis=1)
            y = df["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise Exception(f"Preprocessing Error : {e}")

    def build_model(self, input_dim):
        try:
            model = Sequential([
                tf.keras.Input(shape=(input_dim,)),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            self.model = model
            logging.info("Model Built Successfully")

        except Exception as e:
            raise Exception(f"Model Building Error : {e}")

    def train(self, X_train, y_train):
        try:
            early = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )

            self.history = self.model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early],
                verbose=1
            )

            self.model.save("saved_ann_model.keras")
            logging.info("Model Saved.")

        except Exception as e:
            raise Exception(f"Training Error : {e}")

    def evaluate(self, X_test, y_test):
        try:
            loss, acc = self.model.evaluate(X_test, y_test, verbose=0)

            y_prob = self.model.predict(X_test, verbose=0)
            y_pred = (y_prob > 0.5).astype(int)

            print("\n==============================")
            print("Performance Metrics")
            print("==============================")
            print("Accuracy :", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred))
            print("Recall   :", recall_score(y_test, y_pred))
            print("F1 Score :", f1_score(y_test, y_pred))
            print("ROC AUC  :", roc_auc_score(y_test, y_prob))
            print("Loss     :", loss)

            print("\nClassification Report")
            print(classification_report(y_test, y_pred))

            print("Confusion Matrix")
            print(confusion_matrix(y_test, y_pred))

            self.plot_curves(y_test, y_prob)

        except Exception as e:
            raise Exception(f"Evaluation Error : {e}")

    def plot_curves(self, y_test, y_prob):

        plt.figure(figsize=(8,5))
        plt.plot(self.history.history["accuracy"], label="Train")
        plt.plot(self.history.history["val_accuracy"], label="Validation")
        plt.title("Accuracy")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8,5))
        plt.plot(self.history.history["loss"], label="Train")
        plt.plot(self.history.history["val_loss"], label="Validation")
        plt.title("Loss")
        plt.legend()
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    def run_pipeline(self):

        try:
            logging.info("Pipeline Started")

            df = self.load_dataset()

            X_train, X_test, y_train, y_test = self.preprocess(df)

            self.build_model(X_train.shape[1])

            self.train(X_train, y_train)

            self.evaluate(X_test, y_test)

            logging.info("Pipeline Completed Successfully")

        except Exception as e:
            logging.error(e)
            traceback.print_exc()


if __name__ == "__main__":
    pipeline = BreastCancerANN()
    pipeline.run_pipeline()
