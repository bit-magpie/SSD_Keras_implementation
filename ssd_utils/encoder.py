import numpy as np


class BoxEncoder(object):
    def __init__(self, num_classes, anchors=None, overlap_threshold=0.5):
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = 0 if anchors is None else len(anchors)
        self.overlap_threshold = overlap_threshold

    def _encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned anchors.
        Args
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.
        Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_anchors, 4 + int(return_iou)).
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_anchors = self.anchors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_anchors_center = 0.5 * (assigned_anchors[:, :2] +
                                         assigned_anchors[:, 2:4])
        assigned_anchors_wh = (assigned_anchors[:, 2:4] -
                               assigned_anchors[:, :2])
        
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= assigned_anchors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_anchors[:, -2:]
        return encoded_box.ravel()

    def iou(self, box):
        """Compute intersection over union for the box with all anchors.
        # Args
            box: Box, numpy tensor of shape (4,).
        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_anchors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])
        area_gt *= (self.anchors[:, 3] - self.anchors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def assign_boxes(self, boxes):
        """Assign boxes to anchors for training.
        Args
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.
        Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                anchors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self._encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment
