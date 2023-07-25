import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads message data and categorie data from two filepaths.
  
    Parameters:
    messages_filepath (string): filepath to message data
    categories_filepath (string): filepath to categories data
  
    Returns: pd.DataFrame: dataframe of merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    
    return df

def transform_to_zero(x):
    if isinstance(x, int) and x > 1:
        return 0
    return x

def clean_data(df):
    """
    Cleans categorie data cells, that only binary values are listed
  
    Parameters:
    df (pd.DataFrame): uncleaned dataframe
  
    Returns: pd.DataFrame: cleaned dataframe
    """
    raw_categories = df['categories'].str.split(";", expand=True)
    columns = [raw.split('-')[0] for raw in raw_categories.iloc[0]]
    raw_categories.columns = columns

    raw_categories = raw_categories.applymap(lambda x: x.split('-')[-1])
    raw_categories = raw_categories.astype(int)
    raw_categories = raw_categories.applymap(transform_to_zero)
    
    raw_categories['id'] = df['id']
    df = df.drop('categories', axis=1)
    df = df.merge(raw_categories, on="id")
    
    return df


def save_data(df, database_filename):
    """
    Saves dataframe in a sql table
  
    Parameters:
    df (pd.DataFrame): dataframe with data to be saved
    database_filename (string): database filename
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("mytable", engine, index=False, if_exists='replace')


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