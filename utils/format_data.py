import json
import os
from tqdm import tqdm
import pandas as pd


def save_dt_as_json(mydict, image_name, width, height, debug_dir,   
                    person_confidence_threshold, object_confidence_threshold):
    """Format and save detection in a json file with coco format. Used while porcessing full model inferences."""
    # PATHS
    debug_output_name = image_name.split('.')[0]+".json"
    debug_output_path = os.path.join(debug_dir, 'json')

    detection_boxes = mydict['detection_boxes']
    detection_classes = mydict['detection_classes']
    detection_scores = mydict['detection_scores']

    mydict = {"models": {}}
    all_dtc = []
    for ind, dt_class in enumerate(detection_classes):  # Process all detections
        if (int(dt_class) == 1 and detection_scores[ind] >= person_confidence_threshold) or (int(dt_class)>1 and detection_scores[ind] >= object_confidence_threshold):
            temp_dict = {}
            ymin_norm, ymin = detection_boxes[ind][0], detection_boxes[ind][0]*height
            xmin_norm, xmin = detection_boxes[ind][1], detection_boxes[ind][1]*width
            ymax_norm, ymax = detection_boxes[ind][2], detection_boxes[ind][2]*height
            xmax_norm, xmax = detection_boxes[ind][3], detection_boxes[ind][3]*width
            temp_dict['bbox'] = [float(xmin), float(
                ymin), float(xmax), float(ymax)]
            temp_dict["confidence"] = float(detection_scores[ind])
            temp_dict["class_id"] = float(detection_classes[ind])
            all_dtc.append(temp_dict)

    mydict["models"]['detections'] = all_dtc

    with open(debug_output_path+"/"+debug_output_name, 'w', encoding='utf8') as json_file:
        json.dump(mydict, json_file, indent=1)


def save_dt_as_json_tflite(mydict, image_name, width, height, debug_dir, person_confidence_threshold, object_confidence_threshold):
        """ Format and save detections as a json file. """ 
        json_name = image_name.split('.')[0]+".json"
        json_output_path = os.path.join(debug_dir, 'json', json_name)

        detection_boxes = mydict['detection_boxes']
        detection_classes = mydict['detection_classes']
        detection_scores = mydict['detection_scores']

        all_dt = []
        for ind, dt_box in enumerate(detection_boxes):
            if (int(detection_classes[ind]) == 1 and detection_scores[ind] >= person_confidence_threshold) or (int(detection_classes[ind]>1 and detection_scores[ind] >= object_confidence_threshold)):
                temp_dict = {
                    'ymin': float(dt_box[0])*height,
                    'xmin': float(dt_box[1])*width,
                    'ymax': float(dt_box[2])*height, 
                    'xmax': float(dt_box[3])*width,
                    'confidence': float(detection_scores[ind]),
                    'class': int(detection_classes[ind])}
                all_dt.append(temp_dict)

        json_dict = {"detections": all_dt}
        # Writing to .json
        with open(json_output_path, 'w', encoding ='utf8') as json_file:
            json.dump(json_dict, json_file, indent = 1)


class FormatDataFullModel:
    """Format results from Full Model."""

    def __init__(self, jsons, conf):
        self.jsons = jsons
        self.width = conf['width']
        self.height = conf['height']
        self.prediction_csv = os.path.join(
            conf['debug_output_dir'], 'full_model', 'full_model_debug.csv')
        self.person_confidence_threshold = conf['confidence']['person_confidence_threshold']
        self.object_confidence_threshold = conf['confidence']['object_confidence_threshold']

    def jsons_to_csv(self) -> None:
        """ Convert json files to csv file with the order [image_name, xmin,ymin,xmax,ymax,label]."""
        # Combining jsons into single csv file
        dfs = [self.to_pands(fn) for fn in tqdm(self.jsons)]
        out_dfs = []
        for i in dfs:
            if len(i) != 0:
                out_dfs.append(pd.concat(i, ignore_index=True))
        df = pd.concat(out_dfs, ignore_index=True)
        df['image_name'] = df['image_name'].apply(lambda x: x + '.png')
        df = df.reindex(columns=['image_name', 'xmin',
                                'ymin', 'xmax', 'ymax', 'label', 'confidence'])
        # Storing csv file
        df.to_csv(self.prediction_csv, index=False)

    def to_table(self, d: dict, fn: str) -> pd.DataFrame:
        """ Storing detection values into a DataFrame. """
        return pd.DataFrame(
            {
                "image_name": [fn],
                "xmin": [d["bbox"][0]],
                "ymin": [d["bbox"][1]],
                "xmax": [d["bbox"][2]],
                "ymax": [d["bbox"][3]],
                "confidence": [d["confidence"]],
                "label": [d["class_id"] - 1],
            }
        )

    def read_txt(self, fn: str) -> dict:
        """ Reading the dt json file and storing its conent into a dictionary. """
        with open(fn, 'r') as f:
            contents = json.load(f)
        return contents['models']['detections']

    def to_pands(self, fn: str) -> list:
        """ Convert the dictionary to table. """
        results = self.read_txt(fn)
        # if no detection in the image
        if results == []:
            json_name = str(fn).split('/')[-1]
            image_name = json_name.split('.')[0]
            out = [pd.DataFrame(
                {
                    "image_name": [image_name],
                    "xmin": [''],
                    "ymin": [''],
                    "xmax": [''],
                    "ymax": [''],
                    "confidence": [''],
                    "label": [''],
                }
            )]
        # if at least one detection in the image
        else:
            out = [self.to_table(i, fn=fn.stem) for i in results]
        return out


class FormatDataK210:
    """Formating data output by the k210."""

    def __init__(self, jsons, conf):
        self.jsons = jsons
        self.width = conf['width']
        self.height = conf['height']
        self.prediction_csv = os.path.join(
            conf['debug_output_dir'], 'k210', 'k210_debug.csv')
        self.person_confidence_threshold = conf['confidence']['person_confidence_threshold']
        self.object_confidence_threshold = conf['confidence']['object_confidence_threshold']
        self.count_none = 0

    def jsons_to_csv(self):
        """ Convert json files to csv file while the order [image_name, xmin,ymin,xmax,ymax,label]."""
        # Combining jsons into single csv file
        dfs = [self.to_pands(fn) for fn in tqdm(self.jsons)]
        out_dfs = []
        for i in dfs:
            if len(i) != 0:
                if i[0] is None:
                    self.count_none += 1
                else:
                    out_dfs.append(pd.concat(i, ignore_index=True))
        df = pd.concat(out_dfs, ignore_index=True)
        df['image_name'] = df['image_name'].apply(lambda x: x + '.png')
        df = df.reindex(columns=['image_name', 'xmin',
                                 'ymin', 'xmax', 'ymax', 'label', 'confidence'])
        # Storing csv file
        print("Total None: ", self.count_none)
        df.to_csv(self.prediction_csv, index=False)

    def to_pands(self, fn):
        results = self.read_txt(fn)
        # if no detection in the image
        if results == []:
            json_name = str(fn).split('/')[-1]
            image_name = json_name.split('.')[0]
            out = [pd.DataFrame(
                {
                    "image_name": [image_name],
                    "xmin": [''],
                    "ymin": [''],
                    "xmax": [''],
                    "ymax": [''],
                    "confidence": [''],
                    "label": [''],
                }
            )]
        # if at least one detection in the image
        else:
            out = [self.to_table(i, fn=fn.stem) for i in results]
        return out

    def read_txt(self, fn):
        with open(fn) as f:
            contents = f.readlines()
            contents = [eval(i) for i in contents[1:-1]]
        return contents

    def to_table(self, d, fn):
        w = self.width
        h = self.height
        if (d[0]["class"] == 1) and (d[0]["confidence"] > self.person_confidence_threshold) or (d[0]["class"] > 1) and (d[0]["confidence"] > self.object_confidence_threshold):
            d = d[0]
            d["xmin"] = d["xmin"] * w
            d["xmax"] = d["xmax"] * w
            d["ymin"] = d["ymin"] * h
            d["ymax"] = d["ymax"] * h
            return pd.DataFrame(
                {
                    "xmin": [d["xmin"]],
                    "ymin": [d["ymin"]],
                    "xmax": [d["xmax"]],
                    "ymax": [d["ymax"]],
                    "confidence": [d["confidence"]],
                    "label": [d["class"] - 1],
                    "image_name": [fn],
                }
            )


class FormatDataTFLite:
    """Format results from Full Model."""

    def __init__(self, jsons, conf):
        self.jsons = jsons
        self.width = conf['width']
        self.height = conf['height']
        self.prediction_csv = os.path.join(
            conf['debug_output_dir'], 'tflite', 'tflite_debug.csv')
        self.person_confidence_threshold = conf['confidence']['person_confidence_threshold']
        self.object_confidence_threshold = conf['confidence']['object_confidence_threshold']

    def jsons_to_csv(self) -> None:
        """ Convert json files to csv file with the order [image_name, xmin,ymin,xmax,ymax,label]."""
        # Combining jsons into single csv file
        dfs = [self.to_pands(fn) for fn in tqdm(self.jsons)]
        out_dfs = []
        for i in dfs:
            if len(i) != 0:
                out_dfs.append(pd.concat(i, ignore_index=True))
        df = pd.concat(out_dfs, ignore_index=True)
        df['image_name'] = df['image_name'].apply(lambda x: x + '.png')
        df = df.reindex(columns=['image_name', 'xmin',
                                'ymin', 'xmax', 'ymax', 'label', 'confidence'])
        # Storing csv file
        df.to_csv(self.prediction_csv, index=False)

    def to_table(self, d: dict, fn: str) -> pd.DataFrame:
        """ Storing detection values into a DataFrame. """
        return pd.DataFrame(
            {
                "image_name": [fn],
                "xmin": [d["xmin"]],
                "ymin": [d["ymin"]],
                "xmax": [d["xmax"]],
                "ymax": [d["ymax"]],
                "confidence": [d["confidence"]],
                "label": [d["class"] - 1],
            }
        )

    def read_txt(self, fn: str) -> dict:
        """ Reading the dt json file and storing its conent into a dictionary. """
        with open(fn, 'r') as f:
            contents = json.load(f)
        return contents['detections']

    def to_pands(self, fn: str) -> list:
        """ Convert the dictionary to table. """
        results = self.read_txt(fn)
        # if no detection in the image
        if results == []:
            json_name = str(fn).split('/')[-1]
            image_name = json_name.split('.')[0]
            out = [pd.DataFrame(
                {
                    "image_name": [image_name],
                    "xmin": [''],
                    "ymin": [''],
                    "xmax": [''],
                    "ymax": [''],
                    "confidence": [''],
                    "label": [''],
                }
            )]
        # if at least one detection in the image
        else:
            out = [self.to_table(i, fn=fn.stem) for i in results]
        return out