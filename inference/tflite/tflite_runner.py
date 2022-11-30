import os
import numpy as np
from time import time
from inference.nms import non_max_suppression_relative_fast_class_agnostic, non_max_suppression_relative_fast_multi_class

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TFLiteRunner:
    def __init__(self, path_to_tflite_model, debug=False):
        if debug:
            print("Loading detection model...")
        try:
            import tensorflow as tf
            #self.interpreter = tf.lite.Interpreter(model_path=path_to_tflite_model, num_threads=1)
            self.interpreter = tf.lite.Interpreter(model_path=path_to_tflite_model)
            #self.interpreter.set_num_threads(4)
        except:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(
                    model_path=path_to_tflite_model, num_threads=4)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = False
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        if debug:
            print(self.input_details)
            print(self.output_details)
            print("Detection model loaded!")

    def run_inference_for_single_image(self,
            image_rgb,
            confidence_threshold=0.1,
            use_nms=True,
            nms_iou_threshold=0.4,
            class_agnostic=False,
            debug=False):

        # Extra preprocessing steps to resize and normalize image for network input
        if debug:
            start_time = time()
        if (image_rgb.shape[0] != self.input_height) or (
                image_rgb.shape[1] != self.input_width):
            if debug:
              print("Resizing...")
            #image_rgb = cv2.resize(image_rgb, (self.input_width, self.input_height))
            return 1
            if debug:
                print("\nTime to resize image: {:2f}".format(time() - start_time))

        image_np_expanded = np.expand_dims(image_rgb, axis=0)
        if self.floating_model:
            image_np_expanded = (np.float32(image_np_expanded) - 127.5) / 127.5
        # frame = frame.astype('uint8')

        # Run inference
        if debug:
            start_time = time()
        self.interpreter.set_tensor(self.input_details[0]['index'],
                image_np_expanded)
        self.interpreter.invoke()
        if debug:
            print("\nInference time on the current image: {:2f}".format(
                time() - start_time))

        detected_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        detected_classes = self.interpreter.get_tensor(
                self.output_details[1]['index'])
        detected_scores = self.interpreter.get_tensor(
                self.output_details[2]['index'])

        # Postprocessing; all outputs are numpy arrays, so convert types as
        # appropriate
        detected_boxes = np.squeeze(detected_boxes)
        detected_classes = np.squeeze(detected_classes).astype(np.int64) + 1
        detected_scores = np.squeeze(detected_scores)

        # Filtering out detections with low confidence
        if debug:
            start_time = time()

        indices = np.where(detected_scores > confidence_threshold)
        indices = indices[0]

        if debug:
            print("\nIndices where detection_scores greater than confidence threshold are: ", indices)
            print("\nTime to filter out weak detections: {:2f}".format(
                time() - start_time))

        output_dict = {}

        # Slicing detections with respect to the above indices
        detected_boxes = detected_boxes[indices]
        detected_classes = detected_classes[indices]
        detected_scores = detected_scores[indices]

        if not use_nms:
            # Slicing detections with respect to the above indices
            output_dict['num_detections'] = len(indices)
            output_dict['detection_boxes'] = detected_boxes
            output_dict['detection_classes'] = detected_classes
            output_dict['detection_scores'] = detected_scores

            return output_dict
        # Filtering out overlappting detections with non-maximum suppression
        else:
            if debug:
                start_time = time()
            # Non-maximum supression to get rid of overlapping detections
            if class_agnostic: # Treat all classes as one unique class for faster nms
                detected_boxes, detected_scores, detected_classes = non_max_suppression_relative_fast_class_agnostic(
                        detected_boxes,
                        detected_classes,
                        scores=detected_scores,
                        nms_iou_threshold=nms_iou_threshold)
            else: # Perform nms on each class separately
                detected_boxes, detected_scores, detected_classes = non_max_suppression_relative_fast_multi_class(
                        detected_boxes,
                        detected_classes,
                        detected_scores,
                        nms_iou_threshold)

            if debug:
            #    print("\nIndices after nms: ", indices)
                print("\nTime to perform nms: {:2f}".format(time() - start_time))

            output_dict['detection_scores'] = np.array(detected_scores)
            output_dict['detection_boxes'] = np.array(detected_boxes)
            output_dict['detection_classes'] = np.array(detected_classes)
            output_dict['num_detections'] = len(detected_classes)

            return output_dict