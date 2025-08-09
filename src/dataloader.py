"""
Handles dataset preparation for training, validation, and testing.
"""

import pandas as pd
from pathlib import Path
from tensorflow import keras
import os

def prepare_image_data(image_dir: Path, batch_size: int, target_size=(224, 224), shuffle=True):
    """
    Loads images and labels from a directory into a Keras ImageDataGenerator.
    """
    # Collect all image file paths
    filepaths = list(image_dir.glob("**/*.jpg"))
    
    # Extract label from the parent folder name
    labels = [os.path.split(os.path.split(fp)[0])[1] for fp in filepaths]

    image_df = pd.DataFrame({"Filepath": [str(fp) for fp in filepaths], "Label": labels})

    # Rescale pixel values to [0, 1]
    generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    
    return generator.flow_from_dataframe(
        dataframe=image_df,
        x_col="Filepath",
        y_col="Label",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=shuffle
    )
