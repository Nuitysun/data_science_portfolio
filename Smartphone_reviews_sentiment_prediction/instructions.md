# Technical requirements and implemented scripts description

### Technical requirements 

* Install [Anaconda](https://www.anaconda.com/products/individual#Downloads) distribution
>Note: this project was implemented using Jupiter notebook version 6.0.3 and Python version 3.8.3.
* Install required Python modules via Jupiter notebook:
<pre>
pip install -r requirements.txt
</pre>

### Scripts description 

* **Scraping_reviews.ipynb** scrapes recent reviews data of Smartphones product category from the site https://rozetka.com.ua/. The script can be executed to scrape recent reviews data.
    * Script saves collected data to `.csv` files in `reviews_data` folder with the name of a file `smartphone_reviews _{current_date}.csv`. 

* **Predicting_reviews_sentiment_with_Bert_model.ipynb** contains script to read all `smartphone_reviews_*.csv` files from `reviews_data` folder, manipulate the data, build prediction models and evaluate their performance. 
    * Script saves:
        * `.png` files to `images` folder – images of reviews sentiment are used in README.md. 
        * `data_clean.csv` in `reviews_data` folder - this is the cleaned reviews data which can be used for conducting experiments and training different versions of models in separate scripts. 
    * Script contains commented code to train BERT model - version 1. As training takes a lot of time and computationally expensive, trained model was saved and can be downloaded by link: https://files.fm/u/2msz9eexf – file `sentiment_prediction.data-00000-of-00001`. In order to use the trained model, it’s necessary to download and save it in `/models` folder.

* **Build_Bert_model_v2.ipynb** – contains script to read `data_clean.csv` from `reviews_data` folder, manipulate the data, load and train BERT model – version 2. This model is used in `Predicting_reviews_sentiment_with_Bert_model.ipynb`. 

    * Script contains commented code to train BERT model - version 2. As training takes a lot of time and computationally expensive, trained model was saved and can be downloaded by link: https://files.fm/u/2msz9eexf – file `sentiment_prediction_v2.data-00000-of-00001`. In order to use the trained model, it’s necessary to download and save it in `/models` folder.

>Note: All mentioned `.csv` and `.png` files are provided in repository in `reviews_data` folder.