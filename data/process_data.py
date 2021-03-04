import sys
import pandas as pd
from sqlalchemy import create_engine
from langdetect import detect


def load_data(messages_filepath, categories_filepath):
    """
    Reads the messages and categories csv files from the specified path
    and return the merged dataframe

    Args:
        messages_filepath (str): The filepath of the messages csv file
        categories_filepath (str): The filepath of the categories csv file

    Returns:
        DataFrame: The merged dataframe
    """

    # messages csv file
    messages = pd.read_csv(messages_filepath)

    # categories csv file
    categories = pd.read_csv(categories_filepath)

    # merging messages and categories
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Performs several data cleaning steps

    Args:
        df (DataFrame): The dataframe for which the cleaning steps have to performed

    Returns:
        DataFrame: The cleaned dataframe
    """

    # splitting the string in the categories column into a dataframe
    categories = df['categories'].str.split(';', expand=True)

    # extracting the column names
    row = categories.iloc[0, :]
    category_column_names = list(row.apply(lambda x: x[:-2]))

    # renaming the columns of the new categories df
    categories.columns = category_column_names

    # extracting the integer value from the columns' content
    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)

    # removing the existing categories column in df
    df.drop(['categories'], axis=1, inplace=True)

    # concatenating the df and the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # removing duplicates
    df = df[~df.duplicated(keep='last')]

    # removing non english text
    df = df[df['message'].apply(is_english)]

    # removing rows which have a value of 2 for the related field
    df = df[df.related != 2]

    return df


def is_english(text):
    """
    To detect the language of the text and check if is english

    Args:
        text (str): Any piece of text

    Returns:
        bool: Returns True if english, False otherwise
    """

    try:
        return detect(text) == 'en'
    except:
        return False


def save_data(df, database_filename):
    """
    Saves the given dataframe as a sql table with the given file name

    Args:
        df (DataFrame): The dataframe which needs to be stored to sql
        database_filename (str): The name of the database and the table
    """

    # create the engine and create the database
    engine = create_engine(f'sqlite:///{database_filename}')

    # save the input dataframe into a sql table
    df.to_sql('DisasterResponse', engine, index=False)

    return


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
        print('Please provide the file paths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
