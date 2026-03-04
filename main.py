import numpy as np
import os
import shutil
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata
import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Load training and validation data
train_data = object_detector.DataLoader.from_pascal_voc(
    'gameobjects/train',
    'gameobjects/train',
    ['fuel']
)
val_data = object_detector.DataLoader.from_pascal_voc(
    'gameobjects/valid',
    'gameobjects/valid',
    ['fuel']
)

spec = model_spec.get('efficientdet_lite0')

model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=4,
    train_whole_model=True,
    epochs=12,
    validation_data=None
)

# Export the model locally
model.export(export_dir='.', tflite_filename='celldetector.tflite')
print("\n✅ Model exported successfully as celldetector.tflite!")

# Save to Google Drive so it survives if Colab crashes
drive_path = '/content/drive/MyDrive/celldetector.tflite'
if os.path.exists('/content/drive/MyDrive'):
    shutil.copy('celldetector.tflite', drive_path)
    print(f"✅ Model also saved to Google Drive: {drive_path}")
else:
    print("⚠️ Google Drive not mounted — model only saved locally.")
    print("   Mount Drive first by running: from google.colab import drive; drive.mount('/content/drive')")

# Evaluate model on validation set
print("\n📊 Evaluating model on validation data...")
print(model.evaluate(val_data))
