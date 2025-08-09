"""
Utility functions for the wildfire project.
"""

import json

def save_history_to_json(history, filename):
    """
    Saves training history from Keras to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(history.history, f)



"""
Visualization functions for training history.
"""

import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plots training and validation accuracy/loss.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.show()
