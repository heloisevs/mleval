"""K210 model class """

import os
from pathlib import Path
import tensorflow as tf
from cv2 import cv2
from inference.k210.generate_binaries import make_binaries
from inference.k210.k210_predict import k210_predict


class K210:
    """ Initialization of the k210 model class and processing of the data by it. """

    def __init__(self, model_type, conf):
        self.width = conf['width']
        self.height = conf['height']
        self.path_to_labels = conf['path_to_labels']
        self.images_dir = conf['images_dir']
        self.model_type = model_type
        self.model_path = conf['models'][model_type]
        self.subclasses_file = conf['subclasses_file']

        self.person_confidence_threshold = conf['confidence']['person_confidence_threshold']
        self.object_confidence_threshold = conf['confidence']['object_confidence_threshold']
    
        self.nms_threshold = conf['nms']['nms_threshold']
        self.use_nms = conf['nms']['use_nms']
        self.class_agnostic = conf['nms']['class_agnostic']
        self.debug_dir = os.path.join(conf['debug_output_dir'], model_type)
        self.draw_from_inference = conf['draw_detection']['draw_from_inference']
        self.draw_from_json = conf['draw_detection']['draw_from_json']
        self.draw_from_tf_api = conf['draw_detection']['draw_from_tf_api']
        self.debug = conf['debug']
    
    def process(self):
        """ Pocessing by the k210 requires to create binaries first and then run the inference. """
        # Generate binaries if needed
        model_dir = self.model_path.rsplit('/', 1)[0]
        debug_output_bin_dir = os.path.join(self.debug_dir, 'bin')
        make_binaries(model_dir, self.images_dir, debug_output_bin_dir, self.model_path)
        # Run prediction
        k210_predict(model_dir, self.images_dir, self.debug_dir, self.subclasses_file)
        # Getting all json files
        jsons = list(Path(os.path.join(self.debug_dir, 'json')).glob('*.json'))
        return jsons 