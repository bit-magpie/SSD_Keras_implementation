import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer
import numpy as np


class AnchorBox(Layer):
    """Generate the anchor boxes of designated sizes and aspect ratios.
    Args
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the anchor box coordinates
            such that they are within [0, 1].
    Input shape 4D tensor with shape:
        `(samples, rows, cols, channels)`
    Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)

    """

    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        self.waxis = 2
        self.haxis = 1

        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(AnchorBox, self).__init__(**kwargs)

    def call(self, x, mask=None):
        input_shape = K.int_shape(x)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]

        # define anchor boxes shapes
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # define centers of anchor boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # define xmin, ymin, xmax, ymax of anchor boxes
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
        anchor_boxes[:, ::4] -= box_widths
        anchor_boxes[:, 1::4] -= box_heights
        anchor_boxes[:, 2::4] += box_widths
        anchor_boxes[:, 3::4] += box_heights
        anchor_boxes[:, ::2] /= img_width
        anchor_boxes[:, 1::2] /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)
        if self.clip:
            anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)

        # define variances
        num_boxes = len(anchor_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        anchor_boxes = np.concatenate((anchor_boxes, variances), axis=1)
        anchor_boxes_tensor = K.expand_dims(K.variable(anchor_boxes), 0)
        pattern = [tf.shape(x)[0], 1, 1]
        anchor_boxes_tensor = tf.tile(anchor_boxes_tensor, pattern)

        return anchor_boxes_tensor

    def get_config(self):
        config = {
            'img_height': self.img_size[0],
            'img_width': self.img_size[1],
            'min_size': self.min_size,
            'max_size': self.max_size,
            'aspect_ratios': list(self.aspect_ratios),
            'clip': self.clip,
            'variances': list(self.variances)
        }
        base_config = super(AnchorBox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
