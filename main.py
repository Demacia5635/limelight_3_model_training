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
spec = model_spec.get('efficientdet_lite0')  # much faster, still good

model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=4,
    train_whole_model=True,
    epochs=25,
    validation_data=None   # skip validation during training to save time
)

# ✅ Export the model
model.export(export_dir='.', tflite_filename='celldetector.tflite')

# ✅ Print evaluation results on validation set
print("\n📊 Evaluating model on validation data...")
print(model.evaluate(val_data))

print("\n✅ Model exported successfully as celldetector.tflite!")
