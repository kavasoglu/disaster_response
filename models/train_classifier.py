import sys

import pandas as pd
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle


'''Loads data from a given database.

Args:
    database_filepath: string, the database path for the data to be loaded

Returns:
    X: DataFrame, features dataset
    Y: DataFrame, target dataset
    categories: Index, target class names
'''
def load_data(database_filepath):
    database_path = "sqlite:///" + database_filepath
    engine = create_engine(database_path)
    con = engine.connect()
    df = pd.read_sql('select * from DisasterMessages', con)
    X = df['message']
    Y = df.drop(['message','original','genre'],axis=1)
    categories = Y.columns
    return X,Y,categories


'''Tokenizes given text into words by removing stop words.

Args:
    text: string, text to be tokenized

Returns:
    tokens: list of strings, words after stop words removal and lemmatizer
'''
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return tokens


'''Builds multi output classifier pipeline.

Returns:
    pipeline: Pipeline, multi output classifier pipeline
'''
def build_model():
    return Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=30, random_state=0)))
    ])


'''Evaluates model with the test dataset and prints out the evaluation results.

Args:
    model: Pipeline, machine learning model
    X_test: DataFrame, features test dataset
    Y_test: DataFrame, target test dataset
    category_names: Index, target class names
'''
def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)

    for index, category_name in enumerate(category_names):
        print(index,'-',category_name,':')
        preds = [pred[index] for pred in y_preds]
        print(classification_report(Y_test[category_name], preds))


'''Saves model to a given file path.

Args:
    model: Pipeline, sachine learning model
    model_filepath: string, destination file path for the model to be saved
'''
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


'''Main program of the ML pipeline. Sample usage:
python models/train_classifier.py data/DisasterResponse.db models/multi_classifier.pkl

Args:
    database_filepath: string, the database path for the data to be loaded
    model_filepath: string, destination file path for the model to be saved
'''
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
