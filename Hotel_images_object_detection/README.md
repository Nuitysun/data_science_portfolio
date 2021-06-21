<h1><center><span style="color:#C2571A"> Detect hotel amenities based on rooms photos</span></center></h1>
<h2><center><span style="color:#C2571A"> Computer Vision. Training custom object-detection YoloV5 model</span></center></h1>

## 1. Project Motivation <a name="1"></a>

In this project we will research a possibility of implementing an improvement for a platform with accommodation offers (hotels, apartments, private room renting) which will allow to extend a standard list of amenities provided by a hotel for a specific room. The idea is to process images of a room and detect specific object of interest, for example: swimming pool, desk, kitchen, coffeemaker, TV, bathtub, balcony, garden, etc. This is the first version of amenities to predict and it is to be extended and refined in the next steps.

Results of this project can be useful as a hotel can provide not full information in the list of amenities. It’s also possible that different hotels can have different standards about what to include in the list. 

To improve experience of potential customers who want to book an accommodation, we want to create an extended, searchable and standardized among all hotel list of amenities.  

## 2. The Dataset <a name="2"></a>

For training of custom object-detection YoloV5 model we will use `Open Google Data Set v4`. 
For preliminary evaluation of model results we will use a sample of real hotel photos which will be used in production to predict hotel amenities. 


## 3. Analysis Plan  <a name="3"></a>

The task of this project is supervised machine learning problem that can be solved the best with deep learning methods. 
In this project following steps were performed:
- Define model types that can be used to solve the given task: supervised image Classification and Object detection;
- Assess if existing pre-trained Object detection model can solve the task: check if the list of pre-trained classes contain all necessary classes to be predicted and manually check model predictions on the sample of real hotel photos;
- Assess if existing pre-trained image Classification model can solve the task: check if the list of pre-trained classes contain all necessary classes to be predicted and manually check model predictions on the sample of real hotel photos;
- Define model for custom training (object detection YoloV5 model was chosen);
- Define performance metrics to evaluate custom model based on what is known about expected performance;
- Prepare training dataset, train custom model and evaluate the predictions; 
- Summarize received results and next steps to improve the model.


## 4. Model Evaluation methods   <a name="4"></a> 

### 4.1. Performance metrics for Object detection model

**`Precision`** The higher the precision, the more confident the model is when it classifies a sample as Positive. 

**`Recall`** The higher the recall, the more positive samples the model correctly classified as Positive.

**`Precision-recall curve`** Due to the importance of both precision and recall, a precision-recall curve is used to show the trade-off between the precision and recall values for different thresholds. This curve helps to select the best threshold to maximize both metrics. 

Threshold here refers to predicted probability by the model that object belongs to certain class. When making a prediction model converts probability scores into a class label classification and uses a threshold: when probability score is equal to or above the threshold, the sample is classified as one class. Otherwise, it is classified as the other class (or not classified). 

**`Average precision` (AP)** is a way to summarize the precision-recall curve into a single value representing the average of all precisions. Using a loop that goes through all precisions/recalls, the difference between the current and next recalls is calculated and then multiplied by the current precision. 

**`mAP` (mean Average Precision)** calculates mean among Average Precision of all classes predicted. For our case mAP will be equal to Average Precision since we have only one class to predict.

**`IoU` (Intersection over Union metric)** is a quantitative measure to score how the ground-truth and predicted boxes match. The IoU helps to know if a region has an object or not. The IoU is calculated by dividing the area of intersection between the 2 boxes by the area of their union. The higher the IoU, the better the prediction.  Note that the IoU is 0.0 when there is a 0% overlap between the predicted and ground-truth boxes. The IoU is 1.0 when the 2 boxes fit each other 100%.

To objectively judge whether the model predicted the box location correctly or not, a threshold is used. If the model predicts a box with an IoU score greater than or equal to the threshold, then there is a high overlap between the predicted box and one of the ground-truth boxes. This means the model was able to detect an object successfully. The detected region is classified as Positive (i.e. contains an object).

On the other hand, when the IoU score is smaller than the threshold, then the model made a bad prediction as the predicted box does not overlap with the ground-truth box. This means the detected region is classified as Negative (i.e. does not contain an object).

**`mAP@0.5`** metric refers to Average Precision when IoU is set to 0.5.

**`mAP@0.05:0.95`** metric calculates Average Precision when IoU is set to the range from 0.5 to 0.95. 

More details about described performance metrics for object detection model can be found in this [article](https://www.kdnuggets.com/2021/03/evaluating-object-detection-models-using-mean-average-precision.html).


### 4.2. Evaluation approaches and trade-offs

When evaluating performance of the object-detection model, it’s important to pay attention to two aspects:

1.	Context of the usage. We need to check the performance metrics, but sometimes standard metrics cannot help with certain areas of model evaluation. 

    For example, if IoU (Intersection over Union metric) set to be a threshold of 0.5 and the object detected by the model has overlapping area only of 40%, then such object will not be classified by the model. But how is important for a particular use case accuracy of box limits prediction? For some cases it can be critical, for other cases less important. Note, that some of standard performance metrics for object detection models are calculated for fixed IoU values (`mAP@0.5`, `mAP@0.05:0.95`).

    For our case we need to detect object of interest as accurately as possible and it’s acceptable to have an error margin in box limits, but it shouldn’t be too high as incorrectly defined box limits will lead to errors in predictions of class to witch object belongs. 

2. It’s important to calculate quantitative metrics that can give an objective overview of model performance and allow to compare performance of different models or of the same model before and after tuning. 

    In this project we will calculate the performance metrics for the data obtained from `Open Google Data Set v4` and will evaluate model performance based on this data as this should give the first results of this proof-of-concept project. We will also run custom object detection model on the sample of real hotel photos and will visually explore the results. 

    On the next stage of model development for more reliable model evaluation it’s recommended to select the test data of real hotel photos, label it and calculate performance metrics on this test set. This will allow to objectively evaluate model performance and also give a baseline for comparison during model selection and optimization. 


## 5. Pre-trained models used 

- object detection model YOLOv3 (with [ImageAI library](https://imageai.readthedocs.io/en/latest/));
- image classification model ResNet50 (with [keras.applications module](https://www.tensorflow.org/api_docs/python/tf/keras/applications));
- object detection model YOLOv5 (using [Ultralytics git repository](https://github.com/ultralytics/yolov5)).

# 6. Object detection with pre-trained model YOLOv3

**Model description:** 
The first step of this project will be applying the pre-trained object detection model using ImageAI library. Object detection models seek to identify the presence of relevant objects in images and classify those objects into relevant classes.

In this project we will test object detection of by pre-trained YOLOv3 model. It has  moderate performance and accuracy, with a moderate detection time.

The list of classes on which models were trained can be found [here](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md)

From inspecting the list we can see that some of classes are useful for the purposes of our project, for example: 
- couch,   
- bed,
- dining table,   
- toilet,   
- tv,   
- laptop,  
- microwave,   
- oven,
- toaster,   
- sink,   
- refrigerator,  
- hair dryer.

But also it's important to note that some classes that we want to detect are not available in the pre-trained model (garden, swimming pool, coffeemaker, etc).

**Examples of predictions:**

<table>
  <tr>
    <td valign="top"><img src="./test_images/predictions/new_room_398.jpg"/></td>
    <td valign="top"><img src="./test_images/predictions/new_room_754.jpg"/></td>
  </tr>
</table>


From applying pre-reained model on our sample images we can see that in general object detection of trained classes works. From quick review we can see that predictions for the following classes worked pretty well: bed, dining table, toilet, tv, laptop, sink. There are few undetected objects on the photos, for example a glass table (room_398.jpg) and sink (room_3.jpg).

Overall the results of detection of trained classes are pretty good, but they are limited to only 80 classes in the pre-trained model, which will be not enough to achieve the goal of predicting hotel amenities. Let's explore other pre-trained models with more classes available for prediction.

# 7.  Image Classification with ResNet-50
**Model description:** 
Another approach to solve the given task is to use the image classification model. The model takes images as input and classifies the major object in the image into a set of pre-defined classes.

There are few pre-trained models that we can apply, but here we will try [ResNet-50](https://github.com/onnx/models/tree/master/vision/classification/resnet), which is trained on a million images of 1000 categories from the ImageNet database. 

Here we can approach a task in two ways: 
- use a pre-trained model with 1000 classes;
- perform additional training of the model so it can learn to predict custom classes.

The list of classes that pre-trained model can predict can be reviewed [here](https://github.com/onnx/models/blob/master/vision/classification/synset.txt).

Among 1000 classes in pre-trained model there are few classes that can useful for our use case, for example:
- n03179701 desk
- n04344873 studio couch, day bed
- n03761084 microwave, microwave oven
- n02808440 bathtub, bathing tub, bath, tub
- n04447861 toilet seat
- n04070727 refrigerator, icebox
- n04404412 television, television system
- n09428293 seashore, coast, seacoast, sea-coast

But still some of important classes that we want to predict are missing, for example: swimming pool, coffeemaker, balcony, etc. 

In this section we will test pre-trained Image Classification model ResNet-50 on the sample of real hotel photos.

**Examples of predictions:**

<table>
  <tr>
    <td valign="top"><img src="./test_images/predictions/new_room_398.jpg"/></td>
    <td valign="top"><img src="./test_images/predictions/new_room_754.jpg"/></td>
  </tr>
</table>

<table>
  <tr>
    <td valign="top"><div>Predicted: [('n03201208', 'dining_table', 0.6906476), ('n04070727', 'refrigerator', 0.04460974), ('n02791124', 'barber_chair', 0.021508591), ('n04065272', 'recreational_vehicle', 0.020597309), ('n03761084', 'microwave', 0.018806074)]
</div></td>
    <td valign="top"><div>Predicted: [('n04239074', 'sliding_door', 0.17551132), ('n02788148', 'bannister', 0.17003809), ('n03742115', 'medicine_chest', 0.10629078), ('n04550184', 'wardrobe', 0.09617319), ('n04005630', 'prison', 0.08277177)]</div></td>
  </tr>
</table>

From prediction we can see that some predictions are useful for the purpose of our project, but model was unable to recognize a class 'desk' on all photos and we can see that for almost all useful classes predictions have pretty low confidence as well as there are many not accurate classification results.

Clearly to solve our task we will need to perform additional training ResNet-50 model to improve results for existing classes and train model to classify other classes that we need. 

Since for our sample photos results of object detection model seem to be better, in the next section we will perform custom training for this type of the model. 

# 8. Training of custom object-detection model based on YOLOv5

With pre-trained models we were unable to fully solve the task, both image detection model that we tried in step 6 and image classification model from step 7, don't have some important classes that we need to detect from the photos. 

In this section we will explore how we can train custom object-detection model based on the pre-trained model YOLOv5. Yolo V5 is one of the best available models for Object Detection at the moment. The great thing about this Deep Neural Network is that it is very easy to retrain the network on your own custom dataset.

## 8.1. Prepare data for training

First step in implementing custom object-detection model is to get training data. 

To train the model we will use `Open Images Dataset v4`, this dataset contains **15.4M bounding boxes for 600 object classes** that can be reviewed[here](https://storage.googleapis.com/openimages/2017_07/bbox_labels_vis/bbox_labels_vis.html).

In the dataset we have such useful for us classes as:
- Desk,
- Coffeemaker,
- Swimming pool, etc.

For this project we will train a custom object detection model to detect 'Swimming pool' class. Results can serve as proof-of-concept and it will be possible to extend number of classes for model training if results of this test run is satisfying. 

To create a custom dataset for training and load only necessary data [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit) was used. This tool is useful as it provides an easy to use downloader which allows to filter the data we want to get from `Open Images V4` and stores it in the folder structure. 

## 8.2. Train custom object-detection YOLOv5 model to detect `Swimming pool` class

Training of the model was performed in GoogleColab due to availability of better GPU resources for faster training. The GoogleColab notebook can be accessed via the [link](https://colab.research.google.com/drive/1EfoxoyXrMOWmJo-kiM59FOmHE-9bdw5H?usp=sharing).

In GoogleColab following steps were performed:
0. Archive and upload the Custom Dataset to GoogleDrive.
1. Install Dependencies and clone repository with Yolo v5 model.
2. Download Custom Dataset it the notebook and `data.yaml` file.
3. Data preprocessing. Change format of labels in `txt` files to YOLO format, which is described [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) and change size of images.
    - In this step no additional processing and augmentation  for images was performed. It's possible to explore if model can be improved by applying additional preprocessing steps. 
4. Define Model Configuration and Architecture.
    - In this step we define number of classes for model training (for this test just 1 class) and don't change other standard parameters of the model.
    - The model yolov5l was chosen. More information about YOLO models can be found [here](https://github.com/ultralytics/yolov5). The chosen checkpoint still has pretty fast speed and higher accuracy.
    ![](https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png)
    
5. Train Custom YOLOv5 Detector for 300 epochs.
6. Evaluate Custom YOLOv5 Detector Performance.
    Performance metrics that we got on validation dataset:
    -	Precision and Recall are about 90%+.
    -	`mAP@0.5`(mean Average Precision when IoU is set to 0.5) seems to be to also be pretty high – about 95%.
    -	`mAP@0.05:0.95` (mean Average Precision when IoU is set to the range from 0.5 to 0.95) is expectedly lower, but still reaches almost 80%. For our use case it’s important to detect if the object is in the picture and we care less about precision of box detection, so results from this metric seem to be acceptable. 
7. Run Inference on test set with trained Weights.
    Performance metrics on the test dataset do not differ much:
    - Precision is 96.3%.
    - Recall is 90.8%.
    - `mAP@0.5` is 94.7%.
    - `mAP@0.05:0.95` is 77.4%.
	
## 8.3. Detect `Swimming pool` with custom object-detection YOLOv5 model for hotel photos sample
**Examples of predictions:**
<table>
  <tr>
    <td valign="top"><img src="./yolov5/runs/detect/exp2/room_97.jpg"/></td>
    <td valign="top"><img src="./yolov5/runs/detect/exp2/room_75.jpg"/></td>
  </tr>
</table>

<table>
  <tr>
    <td valign="top"><img src="./yolov5/runs/detect/exp2/room_72.jpg"/></td>
    <td valign="top"><img src="./yolov5/runs/detect/exp2/room_646.jpg"/></td>
  </tr>
</table>

From processed sample images we can see that overall results are pretty good, model was able to detect all pools and boxes detection also seem to be pretty accurate. Even for tricky images model detected objects correctly:
- `room_646.jpg` image contains with river or sea, but model correctly didn't detect a pool in it,
- in `room_72.jpg` image  pool was detected correctly, even though we can see that it is not standard one.
   
For the future steps it's recommended to explore if deeper image preprocessing and augmentation can help to improve results. Also it's important to explore where a model makes mistakes. Intuitively I would guess that differentiating between sea/river/pool/bathtub can be tricky and it's possible that hotel photos can often include both pool and sea/river photos. Possibly it's worth exploring how to overcome this challenge, maybe including seaside photo into training set can help model to learn the difference between two objects.

# 9. Conclusions and Recommendations

### In this project we explored how we can use models with pre-trained classes to predict hotels amenities:
-	 **pre-trained object detection model YOLOv3** pretrained on COCO dataset with 80 classes. 
    - *Pre-trained classes that can be useful for predicting hotel amenities*: couch, bed, dining table, tv, hair dryer, kitchen appliances: microwave, oven, toaster, sink, refrigerator.
    - *Classes that are not available in the pre-trained model and that we want to predict*: swimming pool, desk, coffeemaker, bathtub, balcony, garden.
    - *Results of applying pre-trained model on the sample of real hotel photos*: manual visual check showed that in general object detection of trained classes works pretty well, model was able to detect following classed: bed, dining table, toilet, tv, laptop, sink.
-	**pre-trained image classification model ResNet-50** pretrained on ImageNet database with 1000 classes. 
    - *Pre-trained classes that can be useful for predicting hotel amenities*: desk, studio couch, dining table, bathtub, television, seashore, kitchen appliances: microwave, refrigerator.
    - *Classes that are not available in the pre-trained model and that we want to predict*: swimming pool, coffeemaker, balcony, garden.
    - *Results of applying pre-trained model on the sample of real hotel photos*: manual visual check showed that model didn’t perform well on the sample photos. Some predictions are useful for the purpose of our project, model was able to classify following classes: seashore, studio_couch, television, bathtub, dining_table, refrigerator, microwave. But prediction confidence for the majority if these classes is very low (<20%), model were unable to classify some photos and to recognize a class 'desk' on all photos.

Since ready-to-use models can’t fully solve our task, we performed **custom training of object detection model YOLOv5** that was pre-trained on the COCO dataset. It was decided to perform training for object detection model as this type of pre-trained model provided better results on the sample of real hotel photos.

To train the model we used `Open Images Dataset v4` dataset with annotated 600 classes. This dataset contains following useful for us classes: swimming pool, desk, kitchen appliances, coffeemaker, television, bathtub. Following classes are not found in this dataset: balcony, garden.

### To test the process of custom model training in this project we decided to perform training for only one class and evaluate the results:
-	we downloaded annotated photos for `swimming pool` class using [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit);
-	prepared the data, set model configuration and performed model training for 300 epochs to detect `swimming pool`;
-	results on the validation set: 
    -	Precision and Recall are about 90%+;
    - mAP@0.5 (mean Average Precision when IoU is set to 0.5) is about 95%; 
    - mAP@0.05:0.95 (mean Average Precision when IoU is set to the range from 0.5 to 0.95) is almost 80%. 
-	results on the test set: 
    - Precision is 96.3%;
    - Recall is 90.8%;
    - mAP@0.5 is 94.7%;
    - mAP@0.05:0.95 is 77.4%.
-	manual check of results on the sample of real hotel photos: overall results are pretty good, model was able to detect all pools and boxes detection also seem to be pretty accurate.

## Recommendations on the further steps:
-	Define expected performance expectations of the model for business application;
-	Clarify list of amenities to predict;
-	Research ways to collect the data for model training with all amenities needed (in this project we used `Open Images Dataset v4`, but it doesn’t have data on two classes: balcony, garden): research available annotated datasets that can contain needed classes or ways to annotate custom dataset;
-	Prepare annotated test set (preferably based on real hotel pictures) to calculate metrics to objectively evaluate model performance and calculate baseline performance that can be used in the process of model tuning to systematically estimate model improvements; 
-	Explore if more extensive preprocessing and augmentation of images can improve the results;
-	Explore if hyper-parameters tuning of YOLOv5 model can improve the results;
-	Perform training of YOLOv5 model on extended dataset which contains data with all needed classes and evaluate the results;