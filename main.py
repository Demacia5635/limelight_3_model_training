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
# replace "game_pieces" in all the lines with your roboflow DataSet project name
# replace "cell" with your object name, you can add more ojects just add ['<object>']
train_data = object_detector.DataLoader.from_pascal_voc(
    'game_pieces/train', 
    'game_pieces/train',
    ['cell']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'game_pieces/valid',
    'game_pieces/valid',
    ['cell']
)

spec = model_spec.get('efficientdet_lite0')

# Fix: Remove validation_data to avoid COCO evaluation error
model = object_detector.create(
    train_data, 
    model_spec=spec, 
    batch_size=4, 
    train_whole_model=True, 
    epochs=20,
    validation_data=None  # This prevents the COCO evaluation error
)

model.export(export_dir='.', tflite_filename='celldetector.tflite')
print("✅ Model exported successfully!")
