# Disaster Response Pipeline Project

This project is created to classify disaster messages to help the people who is in need to certain things at the time of disaster as fast as possible.

### Data Source:
Project dataset consisting of disaster messages and categories is provided publicly by [Figure Eight](https://www.figure-eight.com/datasets/).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/multi_classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

### Project Components
1. ETL Pipeline - `data/process_data.py`:
* Loads the messages and categories datasets
* Merges these two datasets
* Cleans the data
* Stores it into a SQLite database.
2. ML Pipeline - `models/train_classifier.py`:
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file
3. Flask Web App `app/` is created to see data visualizations using Plotly.

### Notes:
Table name is given as `DisasterMessages` while storing it into SQLite database.
