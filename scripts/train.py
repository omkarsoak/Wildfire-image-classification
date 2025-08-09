"""
Entry point script for training a wildfire classification model.
"""

from src.config import *
from src.data_loader import prepare_image_data
from src import model_definitions
from src.trainer import train_model
from src.visualizer import plot_training_history

# Load datasets
train_images = prepare_image_data(DATASET_DIR / "train", BATCH_SIZE, IMAGE_SIZE)
val_images = prepare_image_data(DATASET_DIR / "valid", BATCH_SIZE, IMAGE_SIZE)

# Select model
if MODEL_NAME == "vgg16":
    model_class = model_definitions.VGG16Transfer
elif MODEL_NAME == "resnet50":
    model_class = model_definitions.ResNet50Model
else:
    raise ValueError(f"Unsupported model: {MODEL_NAME}")

# Initialize and compile
model_obj = model_class(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
model = model_obj.model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
history = train_model(model, train_images, val_images, MODEL_NAME, EPOCHS)
plot_training_history(history)


"""
Entry point script for evaluating a trained wildfire model.
"""

from src.config import DATASET_DIR, BATCH_SIZE, IMAGE_SIZE, MODEL_NAME
from src.data_loader import prepare_image_data
from src.evaluator import evaluate_and_save

test_images = prepare_image_data(DATASET_DIR / "test", BATCH_SIZE, IMAGE_SIZE, shuffle=False)
evaluate_and_save(f"{MODEL_NAME}_best_model.h5", test_images, MODEL_NAME)
