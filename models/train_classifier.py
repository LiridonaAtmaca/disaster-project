import sys
from sqlalchemy import create_engine
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
<<<<<<< HEAD
    """
    Loads data from a certain database filepath into a pandas dataframe,
    splits it into features (X) and values that need to be predicted (Y) and the
    category names
  
    Parameters:
    database_filepath (string): database filepath
    
    Returns: 
    X, Y, category_names
    """
=======
>>>>>>> 48afdf40523d6ec01abd26d397de39c8342f06b4
    engine = create_engine(f'sqlite:///{database_filepath}')
    sql = 'SELECT * FROM mytable;'
    df = pd.read_sql_table('mytable', engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
<<<<<<< HEAD
    """
    Performs a word tokenize and a lemmatization on the text
  
    Parameters:
    text (string): raw text
    
    Returns: 
    clean_tokens (list): cleaned tokenized words
    """
=======
>>>>>>> 48afdf40523d6ec01abd26d397de39c8342f06b4
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
<<<<<<< HEAD
    """
    Builds a ML pipeline using GridSearch
    
    Returns: cv (object: GridSearchCV): the model
    """
=======
>>>>>>> 48afdf40523d6ec01abd26d397de39c8342f06b4
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__max_depth': [None, 5, 8]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
<<<<<<< HEAD
    """
    Predicts on the test set and prints out a classification report
    
    Parameters:
    model (object): the ml model
    X_test (): test data
    Y_test (): actual true values
    category_names (list): category names
    """
=======
>>>>>>> 48afdf40523d6ec01abd26d397de39c8342f06b4
    y_pred = model.predict(X_test)
    predicted = pd.DataFrame(y_pred, columns = category_names)
    actual = pd.DataFrame(Y_test, columns = category_names)
    
    for column in category_names:
        print(classification_report(actual[column], predicted[column], target_names=[column]))


def save_model(model, model_filepath):
<<<<<<< HEAD
    """
    Saves the model as a pickle file
    
    Parameters:
    model (object): ML model
    model_filepath (string): model filepath
    """
=======
>>>>>>> 48afdf40523d6ec01abd26d397de39c8342f06b4
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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