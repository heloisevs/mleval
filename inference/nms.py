import numpy as np


def non_max_suppression_relative_fast(boxes, scores=None, nms_iou_threshold=0.4):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        print("Expecting coordinates between 0 and 1 for relative bounding boxes!")
        exit()

    # boxes is an np.array
    ymin = boxes[:, 0]
    xmin = boxes[:, 1]
    ymax = boxes[:, 2]
    xmax = boxes[:, 3]

    # Initialize the list of return boxes and scores
    picked_indices = []

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by scores if available, else the bottom-right y-coordinate of the
    # bounding box
    area = (xmax - xmin + 1e-6) * (ymax - ymin + 1e-6)
    idxs = ymax
    if scores is not None:
        idxs = scores
    idxs = np.argsort(idxs)
    #print(idxs)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]

        picked_indices.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(xmin[i], xmin[idxs[:last]])
        yy1 = np.maximum(ymin[i], ymin[idxs[:last]])
        xx2 = np.minimum(xmax[i], xmax[idxs[:last]])
        yy2 = np.minimum(ymax[i], ymax[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1e-6)
        h = np.maximum(0, yy2 - yy1 + 1e-6)

        # Compute the intersection
        intersection = w * h

        # Compute the overlapping IOU ratio between intersection and union
        overlap = intersection / (area[i] + area[idxs[:last]] - intersection)

        # Delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > nms_iou_threshold)[0])))
        #remaining = np.where(overlap < nms_iou_threshold)
        #idxs = idxs[remaining]

    # Return only pick indices
    return picked_indices


def non_max_suppression_relative_fast_class_agnostic(boxes, classes, scores=None, nms_iou_threshold=0.6):
    picked_indices = non_max_suppression_relative_fast(boxes, scores, nms_iou_threshold)

    ret_boxes = boxes[picked_indices]
    ret_scores = scores[picked_indices]
    ret_classes = classes[picked_indices]

    return ret_boxes, ret_scores, ret_classes


def non_max_suppression_relative_fast_multi_class(boxes, classes, scores=None, nms_iou_threshold=0.6):
    class_dict = {}
    for ind, _class in enumerate(classes):
        if _class not in class_dict.keys():
            class_dict[_class] = [ind]
        else:
            class_dict[_class].append(ind)

    ret_boxes = []
    ret_scores = []
    ret_classes = []

    for _class in class_dict.keys():
        _boxes = boxes[class_dict[_class]]
        _scores = scores[class_dict[_class]]

        picked_indices = non_max_suppression_relative_fast(boxes=_boxes, scores=_scores, nms_iou_threshold=nms_iou_threshold)

        ret_boxes.extend(_boxes[picked_indices])
        ret_scores.extend(_scores[picked_indices])
        ret_classes.extend([_class] * len(picked_indices))

    return ret_boxes, ret_scores, ret_classes