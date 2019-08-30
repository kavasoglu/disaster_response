import sys

import pandas as pd
import numpy as np

from sqlalchemy import create_engine


'''Loads and merges the messages and categories data from csv files.

Args:
    messages_filepath: string, csv file path for the messages
    categories_filepath: string, csv file path for the categories

Returns:
    df: DataFrame, merged dataframe of the messages and categories datasets
'''
def load_data(messages_filepath, categories_filepath):
    index_col_name = 'id'
    messages = pd.read_csv(messages_filepath, index_col=index_col_name)
    categories = pd.read_csv(categories_filepath, index_col=index_col_name)
    df = pd.merge(messages, categories, on=index_col_name)
    return df


'''Preprocesses dataframe to extract categories into separate columns.

Args:
    df: DataFrame, dataframe to be preprocessed

Returns:
    df: DataFrame, preprocessed dataframe
'''
def clean_data(df):
    # column names are embedded into column values eg. 'related-1;request-0'
    # extract column names from values
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0,:].values
    category_colnames = [value[0:-2] for value in row]
    categories.columns = category_colnames

    # set each value to be the last character of the string
    # convert column from string to numeric
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1])

    # remove initial categories column since we have the new categories dataframe
    df.drop('categories',axis=1,inplace=True)

    # concat original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicate rows
    df.drop_duplicates(inplace=True)

    return df


'''Saves dataset into a given database.

Args:
    df: DataFrame, dataset to be saved into the database
    database_filename: string, database path to be saved
'''
def save_data(df, database_filename):
    db_path = 'sqlite:///' + database_filename
    engine = create_engine(db_path)
    df.to_sql('DisasterMessages', engine, index=False)


'''Processes messages & categories datasets and saves into a database. Sample usage:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Args:
    messages_filepath: string, csv file path for the messages dataset
    categories_filepath: string, csv file path for the categories dataset
    database_filepath: string, database path for the processed data to be saved
'''
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
