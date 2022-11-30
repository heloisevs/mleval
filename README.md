# ml_models_eval
Repository to evaluate ML models implemented on TensorFlow.

<br>

# Requirements
```
python == 3.7.12
numpy == 1.21.6
cv2 == 4.3.0
tensorflow == 1.15.0
```

# Running instructions
1. Update the "Classes and labels information" category of `config.yml` + provide the corresponding label map (`.pbtxt`) and subclasses (`.txt`) files
1. Place images in the folder `data/images`
2. Place the `<gt_file_name>.csv` file in the folder `data`
3. Run:
```
python main.py -m <model_name> -gt <gt_file_name>.csv
```

|Argument|Type|Description|Required (Y/N)|
|--------|----|-----------|--------|
|`model_name`|string|Must be one of the following: `full_model`, `tflite` or `k210` |Yes|
|`gt_file_name`|`.csv`|Contains ground truth information. It should have 6 columns [`image_name`,`xmin`,`ymin`,`xmax`,`ymax`,`label`] with :<br>* `image_name` a string<br>* `xmin`, `ymin`, `xmax`, `ymax` the absolute coordinates of the gt bounding boxes for a 320x240 image<br>* `label` an integer value starting at `0` |Yes|

## Advanced settings: 
More settings as the confidence and nms thresholds are defined in `config.yml`. 
  
<br />

# Results
This script runs the inference of the selected model on the images (provided in `data/images/`) and then evaluates the Precision, Recall, f1 score, mAP@.5 and mAP@.5:95 for all classes and per class.<br>
* Model's inferences are stored in a csv file in `results/debug/<model_name>`. The file has 7 columns [`image_name`,`xmin`,`ymin`,`xmax`,`ymax`,`label`, `confidence`] with :<br>* `image_name` a string<br>* `xmin`, `ymin`, `xmax`, `ymax` the absolute coordinates of the gt bounding boxes for a 320x240 image<br>* `label` an integer value starting at `0`<br>* `confidence` a float between 0 and 1<br>
* Precision, Recall, f1 score, mAP@.5 and mAP@.5:95 are displayed in the terminal and saved in `results/eval/results.csv`
* f1-Confidence, Precision-Confidence, Precision-Recall and Recall-Confidence curves are saved in `results/eval`

<br />

# Clean the repository
Run the following command to delete results of `model_type`. 
<br>Run without any argument to delete all results (i.e. `results` folder).
```
python clean.py -m <model_name>
```
|Argument|Type|Description|Required (Y/N)|
|--------|----|-----------|--------|
|`model_name`|string|`full_model`, `tflite` or `k210` |No|

<br />

# Examples
A few images and their ground truth are provided in `data/images/examples`.<br> The folder `sample` contains images labelled for 4 classes: Person, Bag, Clothing, Laptop. <br>The folder `sample_2_classes` contains images labeled for 2 classes: Person and Bag.

