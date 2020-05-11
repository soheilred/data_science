import numpy as np
import pandas as pd
import re
import nltk
import sys
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.font_manager

from classifiers import Classifier
from inverted_index import Index


class GenderPredictor(object):
    def __init__(self, inverted_index):
        super().__init__()
        self.inverted_index = inverted_index



def gender_predictor(books_inverted_index, name):
    """Predicts the gender of each name using
        a) closest pronoun
        b) frequency of the pronoun
        in the surrounding sentences of a name in the text books.
    Genders:
        0: Female
        1: Male
    return:
        a list of booleans to the number of methods
    """
    retrieved_docs = books_inverted_index.search(name)
    max_iteration = 5
    num_loops = min(books_inverted_index.get_total_counts(retrieved_docs), max_iteration)
    i = 0
    he_num = 0
    she_num = 0
    distance_to_he = 0
    distance_to_she = 0
    gender = []

    if len(retrieved_docs) == 0:
        return [0, 0]

    doc = retrieved_docs[0]

    for book_id in doc:
        for sentence_id in doc[book_id]:
            if i > num_loops:
                break
            i += 1

            # search for a pronoun using the sentence id and pick the most frequently used pronoun in the sentence and the next one
            # pronoune frequency
            sentence = books_inverted_index.sentences[book_id][sentence_id] + books_inverted_index.sentences[book_id].get(sentence_id + 1)
            he_num += sentence.count('he') \
                        + sentence.count('him') + sentence.count('his')
            she_num += sentence.count('she') \
                        + sentence.count('her') + sentence.count('hers')


            # nearest pronoun
            sentence = books_inverted_index.sentences[book_id][sentence_id].lower()
            name_loc = doc[book_id][sentence_id][0]
            list_distance_to_he = [y - name_loc for y in 
                            [sentence.lower().find(x) for x in ['he', 'his', 'him']]
                                if y - name_loc > -1]
            list_distance_to_she = [y - name_loc for y in 
                            [sentence.lower().find(x) for x in ['she', 'her', 'hers']]
                            if y - name_loc > -1]
            distance_to_he = 999 if len(list_distance_to_he) == 0 else min(list_distance_to_he)
            distance_to_she = 999 if len(list_distance_to_she) == 0 else min(list_distance_to_she)
    
    if he_num > she_num:
        gender.append(1)
    else:
        gender.append(0)

    if distance_to_he < distance_to_she:
        gender.append(1)
    else:
        gender.append(0)

    return gender


def evaluate_gender_prediction(training_gender_df, 
                                test_gender_df, print_flag=False):
    """Evaluate the gender predictors using the rule based predictors
    then train a couple of classifier using the train gender data frame
    and compare the results.
    Genders:
        0: Female
        1: Male
    """
    f1_scores = []
    method_name = ['Frequency', 'Closest']
    # read the test set for obtaining the gender column (response)
    test_set = pd.read_csv("../data/deaths-test.csv")
    test_set.fillna(value=0, inplace=True)
    y_test = test_set['Gender'].values

    print("======= GENDER PREDICTION =======")
    for column in test_gender_df.columns:
        pred = test_gender_df[column]
        # genders = test_gender_df['gender']
        sign_num = int((30 - len(column) - 2 ) / 2)
        f_score = f1_score(y_test, pred)
        f1_scores.append(f_score)
        if print_flag == True:
            print("="*sign_num, column, "="*sign_num)
            print(confusion_matrix(y_test, pred))
            print("f1-score\t: %.4f" % f_score)

    assert(len(f1_scores) == len(method_name))

    # Train a classifier using the features that were previously created 
    # from the text books. A few methods that were previously proven to
    # be working better with the data set are selected.

    # read the training set for obtaining the gender column
    train_set = pd.read_csv("../data/deaths-train.csv")
    train_set.fillna(value=0, inplace=True)
    y_train = train_set['Gender'].values

    cls_scores, cls_mtd_name = self.gender_classifier(training_gender_df,
                                                y_train, test_gender_df,
                                                y_test)

    f1_scores = f1_scores + cls_scores
    method_name = method_name + cls_mtd_name
    self.plot_f1_scores(method_name, 
                        f1_scores,
                        plot_title="Gender Prediction", 
                        file_name='gender_prediction')
    

