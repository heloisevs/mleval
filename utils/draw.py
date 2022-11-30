import os
import json
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from collections import defaultdict



def draw_boxes_on_image(image,
                        output_dict,
                        debug_image_path,
                        person_confidence_threshold,
                        object_confidence_threshold,
                        width, 
                        height
                        ):
    """
    Args:
    image: an array image. Will convert to PIL object.
    output_dict: dictionary with cooridnates of dt boxes, confidence score and classes
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                        (each to be shown on its own line).
    """
    # Drawing configuration
    thickness=1
    colors = {'0': 'Chartreuse', '1': 'Aqua', '2': 'White', '3': 'Blueviolet' }
    
    # Open image with PIL
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    detection_boxes = output_dict['detection_boxes']
    detection_classes = output_dict['detection_classes']
    for ind, detection_score in enumerate(output_dict['detection_scores']):
        if (detection_score > person_confidence_threshold) and (int(detection_classes[ind]) == 1) or (detection_score > object_confidence_threshold) and (int(detection_classes[ind]) > 1):

            ymin = detection_boxes[ind][0]*height
            xmin = detection_boxes[ind][1]*width
            ymax = detection_boxes[ind][2]*height
            xmax = detection_boxes[ind][3]*width

            color = colors[str(output_dict['detection_classes'][ind]-1)]

            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], 
                    width=thickness, 
                    fill=color)
    np.copyto(image, np.array(image_pil))
    cv2.imwrite(debug_image_path, image)


def visualize_images_from_tf_api(output_dict, image_rgb, debug_image_path, path_to_labels, confidence_threshold):
    image_debug = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_debug,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=confidence_threshold,
        line_thickness=2)
    cv2.imwrite(debug_image_path, image_debug)




def draw_boxes_from_json_on_image(  json_path,
                                    image,
                                    debug_image_path,
                                    person_confidence_threshold,
                                    object_confidence_threshold
                                ):
    # Config for drawing
    colors = {'0': 'Chartreuse', '1': 'Aqua', '2': 'White', '3': 'Blueviolet' }
    thickness = 1

    # Read json
    f = open(json_path)
    data = json.load(f)
    f.close()

    # Read image and convert to PIL
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)

    for one_dt in data['models']['detections']:
        detection_boxes = one_dt['bbox'] #list
        detection_score = one_dt['confidence'] #float
        detection_class = one_dt['class_id'] #float
        
        # Draw person detection if score > person threshold or draw object detection if score > object threshold
        if (detection_score > person_confidence_threshold) and (int(detection_class) == 1) or (detection_score > object_confidence_threshold) and (int(detection_class) > 1):

            xmin = detection_boxes[0]
            ymin = detection_boxes[1]
            xmax = detection_boxes[2]
            ymax = detection_boxes[3]

            color = colors[str(int(detection_class)-1)]

            # Draw
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], 
                    width=thickness, 
                    fill=color)

    np.copyto(image, np.array(image_pil))
    cv2.imwrite(debug_image_path, image)

def draw_boxes_from_k210_json_on_image( json_path,
                                        image,
                                        debug_image_path,
                                        person_confidence_threshold,
                                        object_confidence_threshold
                                    ):
    # Config for drawing
    colors = {'0': 'Chartreuse', '1': 'Aqua', '2': 'White', '3': 'Blueviolet' }
    thickness = 1

    # Read json
    with open(json_path) as f:
        data = f.readlines()
        data = [eval(i) for i in data[1:-1]]

    # Read image and convert to PIL
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    width, height = image_pil.size

    draw = ImageDraw.Draw(image_pil)

    for ind, one_dt in enumerate(data):
        one_dt = one_dt[0]
        detection_score = one_dt['confidence']
        detection_class = one_dt['class']

        # Draw person detection if score > person threshold or draw object detection if score > object threshold
        if (detection_score > person_confidence_threshold) and (int(detection_class) == 1) or (detection_score > object_confidence_threshold) and (int(detection_class) > 1):

            ymin = one_dt['ymin'] * height
            xmin = one_dt['xmin'] * width
            ymax = one_dt['ymax'] * height
            xmax = one_dt['xmax'] * width
            confidence = one_dt['confidence']

            color = colors[str(int(one_dt['class'])-1)]

            # Draw
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], 
                    width=thickness, 
                    fill=color)

    np.copyto(image, np.array(image_pil))
    cv2.imwrite(debug_image_path, image)

def draw_boxes_from_tflite_json_on_image(  json_path,
                                    image,
                                    debug_image_path,
                                    person_confidence_threshold,
                                    object_confidence_threshold
                                ):
    # Config for drawing
    colors = {'0': 'Chartreuse', '1': 'Aqua', '2': 'White', '3': 'Blueviolet' }
    thickness = 1

    # Read json
    f = open(json_path)
    data = json.load(f)
    f.close()

    # Read image and convert to PIL
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)

    for one_dt in data['detections']:
        detection_score = one_dt['confidence'] #float
        detection_class = one_dt['class'] #float
        
        # Draw person detection if score > person threshold or draw object detection if score > object threshold
        if (detection_score > person_confidence_threshold) and (int(detection_class) == 1) or (detection_score > object_confidence_threshold) and (int(detection_class) > 1):

            xmin = one_dt['xmin']
            ymin = one_dt['ymin']
            xmax = one_dt['xmax']
            ymax = one_dt['ymax']

            color = colors[str(int(detection_class)-1)]

            # Draw
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], 
                    width=thickness, 
                    fill=color)

    np.copyto(image, np.array(image_pil))
    cv2.imwrite(debug_image_path, image)

def draw_boxes_from_csv(csv_path:str, conf:dict, model_type:str):
    """ Be careful, the gt labels are for 320x240"""
    # Config for drawing
    colors = {'0': 'Chartreuse', '1': 'Aqua', '2': 'White', '3': 'Blueviolet' }
    thickness = 1

    file = open(csv_path, "r")

    gt_boxes = list(csv.reader(file, delimiter=","))
    file.close()

    # Build a dictionary
    gt_boxes.pop(0)
    gt_boxes_dct = {row[0]:[] for row in gt_boxes}
    for i, one_gt in enumerate(gt_boxes): gt_boxes_dct[one_gt[0]].append(one_gt[1:])

    # Create paths
    os.makedirs(os.path.join(conf['debug_output_dir'], model_type, 'images', 'from_csv'), exist_ok=True)
    for image_name in gt_boxes_dct.keys():
        # Set debug path
        debug_image_path = os.path.join(conf['debug_output_dir'], model_type, 'images', 'from_csv', image_name)
        # Open image
        image_path = os.path.join(conf['images_dir'], image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (conf['width'], conf['height']))
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        draw = ImageDraw.Draw(image_pil)
        for ind, gt_box in enumerate(gt_boxes_dct[image_name]):
            xmin, ymin, xmax, ymax, class_id = gt_box
            color = colors[str(int(class_id))]
            (left, right, top, bottom) = (float(xmin), float(xmax), float(ymin), float(ymax))
            draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=thickness, fill=color)
        
        np.copyto(image, np.array(image_pil))
        cv2.imwrite(debug_image_path, image)
