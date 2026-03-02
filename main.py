import numpy as np
import os
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata
import tensorflow as tf

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# ✅ Load training and validation data
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

# ✅ Use efficientdet_lite2 for better accuracy (vs lite0)
spec = model_spec.get('efficientdet_lite2')

# ✅ Train with improved settings
model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=8,           # was 4, bigger batch = better training
    train_whole_model=True,
    epochs=50,              # was 20, more epochs = better accuracy
    validation_data=val_data
)

# ✅ Export the model
model.export(export_dir='.', tflite_filename='celldetector.tflite')

# ✅ Print evaluation results on validation set
print("\n📊 Evaluating model on validation data...")
print(model.evaluate(val_data))

print("\n✅ Model exported successfully as celldetector.tflite!")
