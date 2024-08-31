from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import pickle

from models.ssd import SSD
from ssd_utils.decoder import BoxDecoder


CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"]

IMG_FILE = 'misc/test_image.jpg'

# Initialize model and load weights
model = SSD((300, 300, 3))
model.load_weights('../weights_vgg.hdf5', by_name=True)
decoder = BoxDecoder(21)

inputs = []

# Load image for predictions
img = image.load_img(IMG_FILE, target_size=(300, 300))
img = image.img_to_array(img)
inputs.append(img.copy())
inputs = np.array(inputs, dtype='float32')

preds = model.predict(inputs, batch_size=1, verbose=1)

# Decode prediction to extract confident score, class, and bounding boxes
results = decoder.detection_out(preds)

det_label = results[0][:, 0]
det_conf = results[0][:, 1]
det_xmin = results[0][:, 2]
det_ymin = results[0][:, 3]
det_xmax = results[0][:, 4]
det_ymax = results[0][:, 5]

top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.7]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

# Load image with original size 
img1 = np.array(image.load_img(IMG_FILE))

# Set colors for bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, top_conf.shape[0] + 1)).tolist()
plt.imshow(img1)

# Plot image and bounding boxes
currentAxis = plt.gca()
for i in range(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * img1.shape[1]))
    ymin = int(round(top_ymin[i] * img1.shape[0]))
    xmax = int(round(top_xmax[i] * img1.shape[1]))
    ymax = int(round(top_ymax[i] * img1.shape[0]))
    score = top_conf[i]
    label = int(top_label_indices[i])
    display_txt = '{:0.2f}, {}'.format(score, CLASS_NAMES[label])
    coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
    color = colors[i]
    currentAxis.add_patch(
        plt.Rectangle(
            *coords,
            fill=False,
            edgecolor=color,
            linewidth=2))
    currentAxis.text(
        xmin, ymin, display_txt, bbox={
            'facecolor': color, 'alpha': 0.5})
plt.show()