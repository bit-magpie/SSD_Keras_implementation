from random import shuffle
import numpy as np
from tensorflow.keras.preprocessing import image


class Generator(object):
    """Generate an input to train SSD model
    Args
        gt: groud truth bounding boxes with corresponding classes
        box_encoder: BoxEncoder object to encode boxes
        batch_size: number of sample for one batch
        image_dir: Path of the folder containing training images
        train_imgs: List of names of training images
        val_imgs: List of names of validation images
        image_size: Size of the input image (300,300)
    """

    def __init__(self, gt, box_encoder,
                 batch_size, image_dir,
                 train_imgs, val_imgs, image_size):

        self.gt = gt
        self.batch_size = batch_size
        self.box_encoder = box_encoder
        self.image_dir = image_dir
        self.train_imgs = list(train_imgs)
        self.val_imgs = list(val_imgs)
        self.train_batches = len(train_imgs)
        self.val_batches = len(val_imgs)
        self.image_size = image_size

    def generate(self, train=True):
        """Process images and bounding box cordinates to feed the SSD model
        Args
            train: Boolean value to define whether training or validation
        Returns
            inputs: 4D numpy array (batch_size, 300, 300, 3)
            targets: Numpy array of bounding boxes and corresponding classes
        """
        while True:
            if train:
                shuffle(self.train_imgs)
                keys = self.train_imgs
            else:
                shuffle(self.val_imgs)
                keys = self.val_imgs
            tmp_inputs = []
            tmp_targets = []
            for key in keys:
                img_path = self.image_dir + key
                img = image.load_img(img_path, target_size=self.image_size)
                img = np.array(img).astype('float32')
                y = self.gt[key].copy()

                y = self.box_encoder.assign_boxes(y)
                tmp_inputs.append(img)
                tmp_targets.append(y)

                if len(tmp_targets) == self.batch_size:
                    inputs = np.array(tmp_inputs)
                    targets = np.array(tmp_targets)
                    tmp_inputs = []
                    tmp_targets = []
                    yield inputs, targets
