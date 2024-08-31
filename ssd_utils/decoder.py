import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import placeholder
from tensorflow.compat.v1 import Session


class BoxDecoder(object):
    """Process the prediction
    Args
        num_classes: Number of classes including background
        iou_thresh: Threshold of intersection over union(iou). default=0.45
        max_out_size: Maximum number of bounding boxes
    """

    def __init__(self, num_classes, iou_thresh=0.45, max_out_size=400):
        self.num_classes = num_classes
        self._iou_thresh = iou_thresh
        self._max_out_size = max_out_size
        self.boxes = placeholder(dtype='float32', shape=(None, 4))
        self.scores = placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._max_out_size,
                                                iou_threshold=self._iou_thresh)
        self.sess = Session()

    def _decode_boxes(self, mbox_loc, mbox_anchorbox, variances):
        """Convert bboxes from local predictions to shifted anchors.
        Args
            mbox_loc: Numpy array of predicted locations.
            mbox_anchorbox: Numpy array of anchor boxes.
            variances: Numpy array of variances.
        Return
            decode_bbox: Shifted anchors.
        """
        anchor_width = mbox_anchorbox[:, 2] - mbox_anchorbox[:, 0]
        anchor_height = mbox_anchorbox[:, 3] - mbox_anchorbox[:, 1]
        anchor_center_x = 0.5 * (mbox_anchorbox[:, 2] + mbox_anchorbox[:, 0])
        anchor_center_y = 0.5 * (mbox_anchorbox[:, 3] + mbox_anchorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[:, 0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_width * variances[:, 1]
        decode_bbox_center_y += anchor_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= anchor_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= anchor_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(
            self,
            predictions,
            background_label_id=0,
            keep_max_out_size=200,
            confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.
        Args
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_max_out_size: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.
        Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_anchorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self._decode_boxes(mbox_loc[i],
                                             mbox_anchorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_max_out_size]
        return results
