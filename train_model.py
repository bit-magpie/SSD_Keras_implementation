import pickle
from os import path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from loss_functions.ssd_loss import SSDLoss
from ssd_utils.encoder import BoxEncoder
from ssd_utils.input_generator import Generator
from ssd_utils.downloader import get_dataset
from models.ssd import SSD

IMG_PATH = 'VOCdevkit 3/VOC2007/JPEGImages/'
NUM_CLASSES = 21
BATCH_SIZE = 32
IMG_SIZE = (300, 300)
INPUT_SIZE = (300, 300, 3)

# Load ground truth boxes and default boxes
gt_boxes = pickle.load(open('misc/ground_truth_boxes.pkl', 'rb'))
default_boxes = pickle.load(open('misc/default_boxes.pkl', 'rb'))

# Check whether training data exists, download otherwise
if not path.exists(IMG_PATH):
    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    get_dataset(URL)

# Initialize box encoder
box_encoder = BoxEncoder(NUM_CLASSES, default_boxes)

# Split dataset into training and validation
keys = sorted(gt_boxes.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]

# Initialize the model
model = SSD(INPUT_SIZE, NUM_CLASSES)

model.compile(
    optimizer=Adam(
        lr=0.0003),
    loss=SSDLoss(
        NUM_CLASSES,
        neg_pos_ratio=2.0).compute_loss,
    metrics=['acc'])

# Initialize input generator
generator = Generator(gt_boxes, box_encoder,
                      BATCH_SIZE, IMG_PATH,
                      train_keys, val_keys,
                      IMG_SIZE)

# Initialize tensorboard callback funciton
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format('voc'))

# Set callbacks
callbacks = [
    ModelCheckpoint(
        'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_acc',
        save_best_only=True,
        verbose=1,
        mode='max'),
    tensorboard]

# Start training process
history = model.fit_generator(generator=generator.generate(True),
                              validation_data=generator.generate(False),
                              steps_per_epoch=len(train_keys) // 32,
                              validation_steps=len(val_keys) // 32,
                              epochs=60,
                              verbose=1,
                              callbacks=callbacks)
