import os
import numpy as np
import tensorflow as tf
import cv2
from time import time
from object_detection.utils import ops as utils_ops
# patch tf1 into `utils.ops`
# utils_ops.tf = tf.compat.v1
from inference.nms import non_max_suppression_relative_fast_class_agnostic
from inference.nms import non_max_suppression_relative_fast_multi_class

default_config = tf.ConfigProto()
default_config.gpu_options.per_process_gpu_memory_fraction = 0.5
default_config.gpu_options.allow_growth = True


class TFRunner:
    def __init__(self,
                 path_to_frozen_graph,
                 width,
                 height,
                 config=None,
                 debug=False):

        if debug:
            print("\nLoading detection model...")

        self.width = width
        self.height = height
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                graph_def = tf.GraphDef()
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

        if config is None:
            config = default_config

        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}

            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph(
                    ).get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(
                        detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, self.height, self.width)
                detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.output_tensor_dict = tensor_dict
            self.input_image_tensor = image_tensor

        if debug:
            print("Detection model loaded!")


    def _run(self, image_np_expanded):
        return self.sess.run(
                self.output_tensor_dict, {self.input_image_tensor: image_np_expanded})


    def run_inference_for_single_image(self,
                                       image_rgb,
                                       confidence_threshold=0.4,
                                       use_nms=True,
                                       nms_iou_threshold=0.4,
                                       class_agnostic=False,
                                       debug=False):

        # Extra preprocessing step for models' input
        image_np_expanded = np.expand_dims(image_rgb, axis=0)

        # Run inference
        output_dict = self._run(image_np_expanded)

        # All outputs are float32 numpy arrays, so convert types as appropriate
        detected_boxes = output_dict['detection_boxes'][0]
        detected_classes = output_dict['detection_classes'][0].astype(np.int64)
        detected_scores = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            detected_masks = output_dict['detection_masks'][0]

        # Filtering out detections with low confidence
        indices = np.where(detected_scores > confidence_threshold)
        indices = indices[0]

        if not use_nms:
            output_dict['num_detections'] = len(indices)
            output_dict['detection_scores'] = detected_scores[indices]
            output_dict['detection_boxes'] = detected_boxes[indices]
            output_dict['detection_classes'] = detected_classes[indices]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks_reframed'] = detected_masks[indices]

            return output_dict

        else: # Non-maximum supression to get rid of overlapping detections
            # Slicing detections with respect to the above indices
            detected_boxes = detected_boxes[indices]
            detected_classes = detected_classes[indices]
            detected_scores = detected_scores[indices]
            if 'detection_masks' in output_dict:
                detected_masks = detected_masks[indices]

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

            output_dict['detection_scores'] = np.array(detected_scores)
            output_dict['detection_boxes'] = np.array(detected_boxes)
            output_dict['detection_classes'] = np.array(detected_classes)
            output_dict['num_detections'] = len(detected_classes)

            return output_dict


if __name__ == '__main__':

    PATH_TO_FROZEN_GRAPH = '../../models/signs_of_life/full_model/model_sol.pb'

    # Load the detection model
    detection_model=TFRunner(PATH_TO_FROZEN_GRAPH, debug=True)

    PATH_TO_TEST_IMAGES_DIR='../../data/images'
    image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img_0.png')

    confidence_threshold = 0.1
    use_nms = False
    nms_iou_threshold = 0.4
    class_agnostic = True
    debug = True

    start_time = time()

    for i in range(100):
        image=cv2.imread(image_path, -1)
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_dict=detection_model.run_inference_for_single_image(
            image_rgb,
            confidence_threshold=confidence_threshold,
            use_nms=use_nms,
            nms_iou_threshold=nms_iou_threshold,
            class_agnostic=class_agnostic,
            debug=debug)

    import pprint
    pprint.pprint(output_dict)

    print("\nTotal time: {:2f}".format(time() - start_time))