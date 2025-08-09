"""
Defines the supported deep learning models for wildfire classification.
"""

from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models

class VGG16Transfer:
    """
    VGG16-based transfer learning model.
    """
    def __init__(self, input_shape):
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        
        # Add classification head
        x = layers.Flatten()(base_model.output)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        
        self.model = models.Model(inputs=base_model.input, outputs=output)

class ResNet50Model:
    """
    ResNet50-based transfer learning model.
    """
    def __init__(self, input_shape):
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation="relu")(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        
        self.model = models.Model(inputs=base_model.input, outputs=output)
