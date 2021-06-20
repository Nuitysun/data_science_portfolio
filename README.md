# Data Science Portfolio

This repository contains portfolio of my data science projects presented in the form of Jupyter Notebooks. 

## Contents 

### [Hotel images object detection](Hotel_images_object_detection)

**Problem type:** Computer Vision. Supervised Object detection/Image classification problem. 

**Goal** of the project is to predict hotel amenities (for example: swimming pool, desk, kitchen, coffeemaker, TV, bathtub, balcony, garden) based on the hotel photos. Images processing is handled the best by deep learning technologies. In this project following pre-trained models were tested on real hotel images:
-	object detection model YOLOv3 (with ImageAI library);
-	image classification model ResNet50 (with keras.applications module);
-	object detection model YOLOv5 (using Ultralytics git repository).

Custom object detection model was trained `Open Images Dataset v4` to detect class `Swimming pool`. Following **results** were achieved on the test set:
-	Precision is 96.3%;
-	Recall is 90.8%;
-	mAP@0.5 is 94.7%;
-	mAP@0.05:0.95 is 77.4%.
Custom model was able to detect all swimming pools on the sample of real hotel photos. In the future it’s possible to train custom model to detect all necessary hotel amenities. 

### [Used BMW cars price prediction](Used_BMW_cars_price_prediction)

**Problem type:** Supervised regression task.

The **goals** of this project are:
-	explore what factors affect a sale price of used BMW cars and which characteristics are the most important to determine a car value;
-	build a prototype of model which predicts price of used BMW cars.

Trough data preprocessing, feature engineering, machine learning algorithms testing and hyper-parameters tuning following **results** were achieved on the testing set:
•	explained_variance: 0.9631,
•	mean_absolute_error: 1435.5959,
•	root_mean_squared_error: 2202.7952,
•	mean_squared_log_error: 0.008453.

Then prototype of price predictor for used BMW cars was implemented and applied to predict cars price on real examples from autotrader.co.uk. On real examples we got following results:
•	explained_variance: 0.9341,
•	mean_absolute_error: 628.75,
•	root_mean_squared_error: 961.897,
•	mean_squared_log_error: 0.005877.

### [Smartphone reviews sentiment prediction](Smartphone_reviews_sentiment_prediction)

**Problem type:** NLP. Supervised text classification.

In this project custom web-scraping script was implemented to scrape smartphone review from e-commerce marketplace rozetka.com. The **goal** of this project is to implement predictive model to classify sentiment for reviews without rating. 

For this NLP problem BERT model was applied. Achieved **results** on the testing set:  
•	Accuracy - 98%.
•	Area Under the Receiver Operating Characteristic Curve - 0.97, so there is a 97.7% chance that the model will be able to distinguish between positive and negative class for reviews.
•	Predictions for positive reviews are: almost 97% of all predicted reviews are actually positive and 94% of all positive reviews were detected correctly.
•	Predictions for negative reviews are: 98% of all predicted reviews are actually negative and 99% of all negative reviews were detected correctly.

### [Hotels rates change prediction](Hotels_rates_change_prediction)

**Problem type:** Supervised classification problem with time-series data. 

The **goal** of this project is to predict if at given date hotel changes price for a room (for any stay date of 365 days ahead) so that number of rate shopping iterations can be reduced.  It’s important to predict all changes in rates (recall metric) and in the same time we need to measure precision, as low precision will increase updates iterations. 

The version of the model with increased weight of the positive class (to capture more rate changes) provided following **results:**
•	Lisbon rooms: 90.77% of positive class observations are predicted correctly with 90.77% precision,
•	London rooms: 93.97% of positive class observations are predicted correctly with 93.52% precision,
•	Barcelona rooms: 98.38% of positive class observations are predicted correctly with 70.27% precision.
