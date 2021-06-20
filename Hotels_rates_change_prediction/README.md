# Prediction of daily changes of rates for hotel rooms

## 1. Project motivation

Rate shopping of hotel room rates has computational and monetary cost, yet it is important to have up-to-date information about hotel room rates.

The goal of this project is to test if it’s possible to predict if at a specific day there will be a change in rates for a hotel room using parameters of hotels/rooms and history of rates changes from the past.

The result of this project is a predictive model of hotel room rates changes. Results of this project can be used as a proof of the concept.

##  2. The Dataset

The dataset used in this project contains historical data of rate shopping for the period from 2020-10-29 to 2021-02-24 of hotel rates from web-site providing on-line registration services for different types of accommodation. 

Hotels from our dataset are located in Lisbon, London and Barcelona.
During one rate shopping iteration we get updated rates for each hotel room available for the stay dates for 1 year (365 days) ahead, so one observation is the rate for the room for the given `stay_date`. 

On average there are two rate shopping iteration per day of rates for each room for 365 days of `stay_date`.

Columns in the Dataset:
-  updated_at: object - timestamp of update 
-  stay_date: object - date of stay in hotel room 
- room_id: object 
- rate_current: int64 - rate updated
- rate_old: int64  - rate before update
- cancellaption_policy_current: object 
- cancellation_policy_old: object 
- room_max_occupancy: int64  
- room_name: object 
- hotel_id: object 
- hotel_name: object 
- hotel_city: object 
- hotel_country: object 
- hotel_rating: int64  
- hotel_currency: object 
- hotel_total_room_count: int64  
- min_len_of_stay: int64  
- hotel_longitude: float64
- hotel_latitude: float64    

## 3. Analysis Plan 

The task to be solved in this project is supervised binary classification problem. With the help of a predictive model we will be answering the question: "Will there be at least one change in rates for the whole period (365 days starting from today) for the given hotel room at a specific day (for example, tomorrow)?"

To implement predictive model following steps are to be performed:
- Read the dataset and transform the data:
    - Read the data.
    - Convert columns with dates as datetime and extract datetime features (number of day, week, month, year, week day, etc).
    - Encode binary target variable: `True` if there is at least one change in rates for the room and `False` if rates are not changed.
    - Group the data per update day per room: reflect if there is a change in rates for room for each day.
    - Add feature reflecting distance of hotel to the city center.
- Conduct exploratory data analysis.
- Generate features of history of rate changes (lag features).
- Predict if the rate changes for each day of the next week.
- Summarize the results and outline next steps.

## 4. Performance Metrics

To evaluate performance of the model following metrics will be used:

**Accuracy.** Accuracy is the proportion of true results among the total number of cases examined. This metric is nor robust for class imbalance data, so we will need to keep this in mind. Checking overall accuracy can be useful to evaluate general model performance.

**Recall.** This metric answers the following question: what proportion of actual Positives is correctly classified? This metric is one of the most important for this project as it's important to update all changed rates so we want to capture as many positives as possible.

**Precision.** This metric answers the following question: what proportion of predicted Positives is truly Positive? This metric is important as we want to reduce the number of rates shopping iterations, so we want to update all changed rates, but also to reduce the number of updates through rate shopping.

**AUC: area under the ROC curve.** AUC ROC indicates how well the probabilities from the positive classes are separated from the negative classes. We can use it to evaluate general model performance, but it's important to remember that this metric is sensitive to class imbalance. 

## 5. Read the dataset and transform the data

### 5.1 Read the data and check descriptive statistics
For now it's decided to drop 'room_name'/'hotel_name', but it’s possible that we can to extract useful information from these columns. For example, word ‘Economy’ in the name of the room can be a useful feature that has an affect a pattern of rate changes. 

### 5.2  Convert columns with dates as datetime and extract datetime features. Encode target variable.

### 5.3 Group the data per day per room: reflect if there is a change in rates for room for update day

###  5.4 Add feature `distance_to_center`
We will use `hotel_longitude` and `hotel_latitude` features to calculate distance of hotels to the city center.

## 6. Exploratory data analysis  

## 7. Generate features of price changes history (lag features)
Frequency of rate changes from the previous periods should be useful to predict if rate will change in the future. Here we will add to each observation history of rate changes (for room and hotel) from the past periods.

## 8. Predict if rate changes for each day of the next week

### 8.1 Define a function for model evaluation
To evaluate model ability to predict if there is change in rates we will take the last week as test data and will try to predict the change in rates for the each day of the next week based on the data we currently have.

Below is the function to test model performance that will:
- filter out data the last 7 days to be our test set,
- change in the main dataframe target for the last 7 days to be equal to 0.5 (to avoid information leakage during features generation),
- generate history (lag) features,
- split the data to training and test sets,
- train the model and make predictions,
- print performance metrics.

### 8.2 Build separate models for the each city and evaluate performance 
On the EDA stage we could see from charts plotted on the initial dataset that data for Lisbon, London and Barcelona data seem to vary a lot. So here we will initialize separate models to predict rate changes for the each city.

**Predictions for day `Lisbon` data:** 

Results of training set:
- Average overall accuracy (for both classes) on training set: 97.51%
Results on testing set:
- Average overall accuracy (for both classes): 88.79%
- Average recall for positive class (price changed): 84.23% of rooms with changed prices were correctly identified.
- Average precision for positive class (price changed): 94.81% of rooms that were predicted to have changed prices actually had changes in prices.
- roc_auc_score: 89.30

For the test week there were 473 observations. Model predicted that we need to update 231 rooms (48.84% of all rooms) and this way we capture 84.23% of all rates changes.

**Predictions for day `London` data:** 

Results of training set:
- Average overall accuracy (for both classes) on training set: 96.73%
Results on testing set:
- Average overall accuracy (for both classes): 90.12%
- Average recall for positive class (price changed): 93.97% of rooms with changed prices were correctly identified.
- Average precision for positive class (price changed): 93.52% of rooms that were predicted to have changed prices actually had changes in prices.
- roc_auc_score: 84.93

For the test week there were 800 observations. Model predicted that we need to update 633 rooms (79.12% of all rooms) and this way we capture 93.97% of all rates changes.

**Predictions for day `Barcelona` data:** 
Results of training set:
- Average overall accuracy (for both classes) on training set: 90.89%
Results on testing set:
- Average overall accuracy (for both classes): 69.73%
- Average recall for positive class (price changed): 79.60% of rooms with changed prices were correctly identified.
- Average precision for positive class (price changed): 73.37% of rooms that were predicted to have changed prices actually had changes in prices.
- roc_auc_score: 66.81

For the test week there were 806 observations. Model predicted that we need to update 537 rooms (66.63% of all rooms) and this way we capture 79.60% of all rates changes.

### 8.2 Increase weight for positive class prediction as it's important to capture all rates changes
We can improve recall for positive class by sacrificing precision of predictions.

**Predictions for day `Lisbon` data with `scale_pos_weight` = 2.5:** 
- Average recall for positive class (price changed) increased to 90.77% (from 84.23%), average precision reduced to 90.77% (from 94.81%)

**Predictions for day `Barcelona` data with `scale_pos_weight` = 2.5:** 
- Average recall for positive class (price changed) increased to 98.38% (from 79.60%), average precision reduced to 70.27% (from 73.37%)

## 9. Conclusions and Recommendations

The goal of this project was to build a model to predict if at a given day there will be at least one change in rates for hotel room for 365 days ahead.

To implement the model, we grouped the data by `update_day` and `room_id`, fixed some vales in rates values and extracted features for the modeling. During the EDA we saw that there seem to be a significant difference in the patterns for rates updates for different hotels/rooms. We also could see that average values of target variable changed drastically in time.

To test the model performance, we took the last 7 days of the dataset and predicted rate change per room for the whole week ahead without updating model with the recent data. 

**The version of the model with increased weight of the positive class provided following results:**
- Lisbon rooms: 90.77% of positive class observations are predicted correctly with 90.77% precision,
- London rooms: 93.97% of positive class observations are predicted correctly with 93.52% precision,
- Barcelona rooms: 98.38% of positive class observations are predicted correctly with 70.27% precision.

**Limitations of the price prediction model:**
1. From the grouped data we can see that the share of observations with changed rates is pretty high (from 68% to 77%), so this is a limitation on how using predictive a model can reduce amount of scarping. If it's possible to achieve high accuracy of model predictions, judging from the data it will allow to reduce rate shopping iterations by 32-23% at maximum. The first steps to reduce scarping could be:
   - to perform rate shopping only 1 time per day for rooms for which changes are not expected as this day. The accuracy of predictions will be a bit higher if the model has recent data of rate changes from the previous days.
   - it's worth considering how we can to optimize rate shopping by `stay date`. For example, if for a specific city/hotel/room for specific stay days probability of rate change is 10% or less - we can skip rate shopping of this stay dates.

2. If we start to use the model to reduce the number of rate shopping iterations, it means that we will have less data for updating model and accuracy of predictions may decrease. It's important to think through strategies of keeping collecting sufficient amount of the data to update the model and keep accuracy at required level and in the same time we want to reduce number of rate updates to the optimal level. During model testing we predicted rate changes for each day for 7 days ahead, so for more distant days model didn't have recent data with rate changes history – it was done on purpose to estimate how well model can make prediction with less data.
3. As part of this project I also tested prediction of rate changes only for bookings with `stay_date` more than 45 days in the future for Lisbon rooms. Accuracy of predictions was quite low (about 65%), it seems like there is too much randomness in rates update patterns for booking for distant future `stay_date`.
4.	It’s necessary to define what are minimal requirements for the model performance and estimate if it’s feasible to achieve them.
5.	CatBoostRegressor algorithm provided the best results on the test set among several algorithms that were tested, but amount of data to be processed and the expected execution time should be taken into consideration when applying price prediction model. It's possible to test other faster algorithms if it's a critical factor.

**Possible ways to improve performance of the model:**

1. In current project feature engineering and selection was performed for Lisbon data. It's possible to explore if additional features can improve predictions for London and Barcelona data.
2. Model that was fit to Barcelona data overfits, it's worth exploring how to make this model generalize better. 
3. Explore hyperparameter tuning for CatBoostRegressor. 
4. Test other machine learning algorithms.
5. Extract more features from the data (room names, history of rate changes, mark days with announcement of changes in COVID restrictions, etc).
6. Collect more information about the hotels (for example, if a hotel belongs to a chain, etc).
7. Collect more historical data (for example, there should be seasonality in the data, so patterns of rate changes from the same season last year could be useful).
