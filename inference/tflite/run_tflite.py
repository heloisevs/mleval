"""Tflite model class """

import os
from pathlib import Path
import tensorflow as tf
from cv2 import cv2
from inference.tflite.tflite_runner import TFLiteRunner
from utils.format_data import save_dt_as_json_tflite
from utils.draw import draw_boxes_on_image, draw_boxes_from_tflite_json_on_image, visualize_images_from_tf_api


class TFLite:
    """ Initialization of the tflite model class and processing of the data by it. """

    def __init__(self, model_type, conf):
        self.model_type = model_type
        self.model_path = conf['models'][model_type]
        self.width = conf['width']
        self.height = conf['height']
        self.path_to_labels = conf['path_to_labels']
        self.images_dir = conf['images_dir']

        self.person_confidence_threshold = conf['confidence']['person_confidence_threshold']
        self.object_confidence_threshold = conf['confidence']['object_confidence_threshold']
        self.inference_tf_models_confidence_threshold = conf['confidence']['tf_models_inference']
    
        self.nms_threshold = conf['nms']['nms_threshold']
        self.use_nms = conf['nms']['use_nms']
        self.class_agnostic = conf['nms']['class_agnostic']
        self.debug_dir = os.path.join(conf['debug_output_dir'], model_type)
        self.draw_from_inference = conf['draw_detection']['draw_from_inference']
        self.draw_from_json = conf['draw_detection']['draw_from_json']
        self.draw_from_tf_api = conf['draw_detection']['draw_from_tf_api']
        self.debug = conf['debug']
    

    def process(self):
        """Running TFLITE models with TFliteRunner. """
        images = os.listdir(self.images_dir)

        # Create json debug directory
        debug_json_output_dir = os.path.join(self.debug_dir, 'json')
        os.makedirs(debug_json_output_dir, exist_ok=True)

        # Loading model
        tflite_model = TFLiteRunner(self.model_path, debug=self.debug)
        for image_name in images :
            image = cv2.imread(os.path.join(self.images_dir, image_name))
            # Check if the size of the image is (320, 240), resize otherwise
            (img_width, img_height, img_channels) = image.shape
            if (img_width != self.width) or (img_height != self.height):
                print("Image size is ", image.shape, ". Resizing to 320 x 240.")
                image = cv2.resize(image, (self.width, self.height))
            # Processing the image
            output_dict = tflite_model.run_inference_for_single_image(
                    image,
                    confidence_threshold= self.inference_tf_models_confidence_threshold,
                    use_nms=self.use_nms,
                    nms_iou_threshold=self.nms_threshold,
                    class_agnostic=self.class_agnostic,
                    debug=False)
            # Saving data to json files and create dt debug image
            save_dt_as_json_tflite(output_dict, image_name,   self.width, self.height,
                                    self.debug_dir, self.person_confidence_threshold, self.object_confidence_threshold)
            
            # Debug purposes
            self.get_debug_images(image, image_name, output_dict, 
                                    draw_from_inference=self.draw_from_inference,
                                    draw_from_json=self.draw_from_json, 
                                    draw_from_tf_api=self.draw_from_tf_api)
    
        
        # Getting paths of all json files
        jsons = list(Path(os.path.join(self.debug_dir, 'json')).glob('*.json'))
        return jsons


    def get_debug_images(self, image, image_name, output_dict, draw_from_inference=False,
                        draw_from_json=False, draw_from_tf_api=False):
        # Initiate paths
        debug_json_output_dir = os.path.join(self.debug_dir, 'json')
        debug_output_name = image_name.split('.')[0]+".json"
        debug_json_path = os.path.join(
            debug_json_output_dir, debug_output_name)

        # Draw bb dt read from the inferece
        image = cv2.resize(image, (self.width, self.height))
        if draw_from_inference:
            # Verify that debug folder exists
            debug_image_dir = os.path.join(
                self.debug_dir, 'images', 'inference')
            os.makedirs(debug_image_dir, exist_ok=True)
            debug_image_path = os.path.join(debug_image_dir, image_name)
            # Draw
            draw_boxes_on_image(image, output_dict, debug_image_path,
                                self.person_confidence_threshold,
                                self.object_confidence_threshold,
                                self.width, self.height)
        # Draw bb dt read fro the json
        if draw_from_json:
            # Verify that debug folder exists
            debug_image_dir = os.path.join(
                self.debug_dir, 'images', 'from_json')
            os.makedirs(debug_image_dir, exist_ok=True)
            debug_image_path = os.path.join(debug_image_dir, image_name)
            # Draw
            draw_boxes_from_tflite_json_on_image(debug_json_path, image, debug_image_path,
                                        self.person_confidence_threshold, self.object_confidence_threshold)
        # Draw bb dt from TF API
        if draw_from_tf_api:
            # Verify that debug folder exists
            debug_image_dir = os.path.join(self.debug_dir, 'images', 'tf_api')
            os.makedirs(debug_image_dir, exist_ok=True)
            debug_image_path = os.path.join(debug_image_dir, image_name)
            # Draw
            visualize_images_from_tf_api(output_dict, image, debug_image_path,
                                        self.path_to_labels, self.object_confidence_threshold)
