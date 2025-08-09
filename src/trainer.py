"""
Handles training of deep learning models with callbacks.
"""

import tensorflow as tf
from src.utils import save_history_to_json
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json


def train_model(model, train_data, val_data, model_name, epochs):
    """
    Compiles and trains the model, saving training history as JSON.
    """
    best_loss = float("inf")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3),
        tf.keras.callbacks.ModelCheckpoint(f"{model_name}_best_model.h5", save_best_only=True)
    ]

    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)
    save_history_to_json(history, f"{model_name}_train_history.json")
    
    return history


"""
Evaluates trained models and produces a confusion matrix with metrics.
"""
def evaluate_and_save(model_path, test_images, model_name):
    """
    Evaluates model and saves metrics to JSON.
    """
    model = load_model(model_path)
    results = model.evaluate(test_images)

    predictions = (model.predict(test_images) >= 0.5).astype(int)
    cm = confusion_matrix(test_images.labels, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    metrics = {
        "Loss": float(results[0]),
        "Accuracy": float(results[1])
    }
    with open(f"{model_name}_test_results.json", "w") as f:
        json.dump(metrics, f)
    
    return results

