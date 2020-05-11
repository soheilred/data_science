import numpy as np
import pandas as pd
import re
import nltk
import sys
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import multiprocessing 

from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer
import matplotlib.font_manager

from classifiers import Classifier
from inverted_index import Index
from network import Graph

deadly_list = ['die', 'kill', 'chop', 'stab', 'strangle', 'assassinate','drown','execute','get','hit','massacre','murder','poison','slaughter','slay','annihilate','asphyxiate','crucify','dispatch','dump','electrocute','eradicate','erase','exterminate','extirpate','finish','garrote','guillotine','hang','immolate','liquidate','lynch','neutralize','obliterate', 'suffocate', 'waste', 'zap', 'sacrifice', 'smother', 'snuff']


def name_freq_to_features(self, features, name_counts):
    """
    turn the name counts into a feature column
    """
    total_characters_count = sum(list(name_counts.values()))
    names_df = features['Name'].str.lower()
    names = [re.sub("[()]", "", x) for x in names_df]
    freqs = [name_counts[name] for name in names]
    add_feature(features)



def fake_score_generator(dataset):
    # generating the file
    scores = pd.DataFrame(0.5, index = dataset.index, columns=["scores"])
    scores.to_csv("../output/fake_scores.csv")
    # reading and loading the file
    fake_scores = pd.read_csv("../data/fake_scores.csv", header=0, names=['scores'])
    return fake_scores



def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


def extract_names_nltk(sentences):
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

    entity_names = []
    for tree in chunked_sentences:
        # Print results per sentence
        # print extract_entity_names(tree)
        entity_names.extend(extract_entity_names(tree))
    # Print all entity names
    #print entity_names
    # Print unique entity names
    print(set(entity_names))


def create_name_dict(names_list):
    id = 0
    names_dict = {}
    for name in names_list:
        full_name = name.split()
        for name in full_name:
            if name in names_dict:
                names_dict[name].append(id)
            else:
                names_dict[name] = [id]
        id += 1
    # testing the dictionary of names
    # print(names_dict['donnel'])
    # print(names_dict['jon'])
    # print(names_dict['davos'])
    return names_dict
    # entities = nltk.chunk.ne_chunk(tagged)
    # for sentence in sentences:


def get_file_names(filename):
    if filename == 'prediction':
        df = pd.read_csv("../data/character-predictions.csv")
        names_df = df['name'].copy()

    if filename == 'death':
        df = pd.read_csv("../data/character-deaths.csv")
        names_df = df['Name'].copy()
    # names_df = df['Name'].str.lower()
    # names = [re.sub("[()]", "", x) for x in names_df]

    print(names_df.shape[0], "names in the dataset")
    return names_df


def create_features(names_list, books_inverted_index, graph,
                    count_flag, gender_flag, 
                    proximity_flag, tf_idf_flag, graph_flag):
    """Creates the set of features, depending on the flags that are set in 
    the main by the command line.

    Arguments:
        names_list {list} -- List of names we want to make the feature set for
        books_inverted_index {inverted index} -- Books' inverted index
        graph {graph} -- Graph object of networkx library
        count_flag {boolean} -- Flag for mention count features
        gender_flag {boolean} -- Flag for gender features
        proximity_flag {boolean} -- Flag for proximity to deathly words feature
        tf_idf_flag {boolean} -- Flag for tf-idf feature (time consuming)
        graph_flag {boolean} -- Flag for graph features (time consuming)

    Returns:
        list -- a list of features
    """    

    features = []
    indices = [0]
    if count_flag:
        feature0 = create_total_count_feature(names_list, books_inverted_index)
        feature1 = create_count_per_book_feature(names_list,
                                                books_inverted_index)
        features.append(feature0)
        features.append(feature1)
        indices.append(len(features))

    if gender_flag:
        feature2 = create_gender_feature(names_list, books_inverted_index)
        features.append(feature2)
        indices.append(len(features))

    if proximity_flag:
        feature3 = proximity_deadly_words(names_list,books_inverted_index)
        if tf_idf_flag:
            feature4 = tf_idf_proximity_deadly_word(names_list,
                                                    books_inverted_index)
            features.append(feature4)

        feature5 = count_deadly_last_n_mentions(names_list,
                                                books_inverted_index,n=5)
        feature6 = count_deadly_last_n_mentions(names_list,
                                                books_inverted_index,n=1)
        features.append(feature3)
        features.append(feature5)
        features.append(feature6)
        indices.append(len(features))

    if graph_flag:
        feature7 = graph.get_graph_features(names_list, 
                        books_inverted_index, 
                        method='characters_in_same_sentence')

        features.append(feature7)
        indices.append(len(features))

    return indices, features


def best_subset_selection(classifier, training_features, test_features):
    """Runs Backward Best Subset Selection on the set of features. Removes 
    one subset, calculates f1-score, and selects the subset that is associated
    with the highest f1-score. This subset has to be removed from the whole set.

    Arguments:
        classifier {classifier} -- a classifier object
        training_features {list} -- a list of features for the training set
        test_features {list} -- a list of features for the test set
    """    
    cur_training_features = []
    cur_test_features = []
    all_indices = list(range(len(training_features)))
    remaining_indices = {i:0 for i in range(len(training_features))}
    fscores_list = []
    removed_index_list = []
    removed_index_fscore = []

    for i in range(len(training_features)):
        for ind in remaining_indices:
            cur_training_features = [training_features[indice] 
                                    for indice in remaining_indices
                                    if indice is not ind]

            cur_test_features = [test_features[indice] 
                                for indice in remaining_indices
                                if indice is not ind]

            classifier.set_features(cur_training_features, cur_test_features)
            classifier.decision_tree()
            fscore, _ = classifier.get_f1scores()
            # fscores_list.append(fscore)
            remaining_indices[ind] = fscore

        index_to_remove = max(remaining_indices, key=remaining_indices.get)
        fscores_list.append(remaining_indices.copy())
        removed_index_fscore.append(remaining_indices.pop(index_to_remove)[0])
        removed_index_list.append(index_to_remove)

    print(fscores_list)
    print(removed_index_list)
    print(removed_index_fscore)

def ablation_test(classifier, indices, training_features, test_features):
    """Implements ablation test, removes one set of feature at a time, and 
    calculates the fscore with respect to the remaining features. Plot them
    at the end, and save it.
    
    Arguments:
        classifier {Classifier} -- The object of classifier
        indices {list} -- the indecies of each feature set with respect to the
        whole feature set
        training_features {list} -- a list of training features set
        test_features {list} -- a list of test features set
    """    
    cur_training_features = []
    cur_test_features = []
    all_indices = list(range(len(training_features)))
    fscores_list = []
    method_names = []

    # cleaning the f1-score list in the classifier before filling it up
    classifier.get_f1scores()
    for i in range(len(indices) - 1):
        removed_indices = list(range(indices[i], indices[i+1]))
        target_indices = [ind for ind in all_indices if 
                                ind not in removed_indices]
        # print(target_indices)

        cur_training_features = [training_features[indice] for 
                                indice in target_indices]
        cur_test_features = [test_features[indice] for 
                            indice in target_indices]

        classifier.set_features(cur_training_features, cur_test_features)

        classifier.logistic_regression()
        # classifier.svc_polynomial()
        # classifier.svc_guassian_kernel()
        classifier.svc_sigmoid()
        classifier.decision_tree()
        classifier.k_nearest_neighbors()
        classifier.naive_base()
        fscore, method_names = classifier.get_f1scores()
        fscores_list.append(fscore)

    plot_ablation(method_names, fscores_list, 'Ablation Test', 'ablation')

    
def plot_ablation(method_name, f1_scores, plot_title, file_name):
    """Plots the result of ablation test.

    Arguments:
        method_name {list} -- list of all ml methods tried on the data set
        f1_scores {list} -- a list of resulting f1-scores
        plot_title {string} -- the title of the plot
        file_name {string} -- the name of a plot to be saved
    """    
    matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') 
    plt.rcParams['font.family'] = "Comfortaa" # font.fantasy # sans-serif

    # plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (8,4)

    x_pos = np.arange(len(method_name))
    width = 0.18 # .25 

    fig, ax = plt.subplots()
    plt.grid(color='w', alpha=.35, linestyle='--')
    ax.patch.set_facecolor(color='gray')
    ax.patch.set_alpha(.35)
    i = 0

    for exp_f_score in f1_scores:
        rect = ax.bar(x_pos + ((i - 1.5) * width), exp_f_score,
                        align='center', width=width, alpha=0.9,
                        label='Feature set {} removed'.format(i))
        autolabel(rects=rect, ax=ax)
        i += 1

    # ax.set_xticks(x_pos, method_name)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_name)
    ax.set_ylim([0,1])
    plt.xlabel('Methods')
    plt.ylabel('F1-Score')
    ax.set_title(plot_title)
    ax.legend() # we can add more measures to this plot
    plt.tight_layout()

    plt.savefig('../output/' + file_name + '.png')
    # plt.show()
    plt.close()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=6)


def create_total_count_feature(names_list, books_inverted_index):
    name_counts = []
    for name in names_list:
        retrieved_query = books_inverted_index.search(name)
        name_counts.append(books_inverted_index.get_total_counts(retrieved_query))
    
    # print(name_counts)
    name_counts_df = pd.DataFrame(name_counts, columns=['total_count'])
    # name_counts_df.to_csv("../output/total_count.csv")
    return name_counts_df


def create_count_per_book_feature(names_list, books_inverted_index):
    name_counts = []
    for name in names_list:
        retrieved_query = books_inverted_index.search(name)
        name_counts.append(books_inverted_index.get_counts_per_book(retrieved_query))
    
    # print(name_counts)
    counts_per_book = pd.DataFrame(name_counts, columns=['b1_count', 'b2_count', 'b3_count', 'b4_count', 'b5_count'])
    # counts_per_book.to_csv("../output/count_per_book.csv")
    return counts_per_book


def create_gender_feature(names_list, books_inverted_index):
    gender = []
    for name in names_list:
        gender.append(gender_predictor(books_inverted_index, name))
    gender_df = pd.DataFrame(gender, columns=['gender_freq', 'gender_proximity'])
    # gender_df.to_csv("../output/genders.csv")
    return gender_df


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


def create_deadly_dict(books_inverted_index):
    """Creates a dictionary of deadly words occured in the books, based on a list of deadly words.
    return:
        deadly words occurance dict (dict of 5)
    """

    deadly_sentences = []
    deadly_dict = {}

    for deadly_word in deadly_list:
        retrieved_docs = books_inverted_index.search(deadly_word)
        if len(retrieved_docs) > 0:
            deadly_sentences.append(retrieved_docs[0])
    
    # combine all the results together; merge the retrieved documents into one dict
    for result in deadly_sentences:
        for book_num in range(5):
            if deadly_dict.get(book_num) is None:
                deadly_dict[book_num] = set()
            # add retrieved sentences for each word to the set of deadly sentences
            if result.get(book_num) is not None:
                deadly_dict[book_num].update(result[book_num])

    # print(deadly_dict)
    return deadly_dict


def proximity_deadly_words(names_list, books_inverted_index):
    """Figures out if a name in the name_list is in the same sentence as a
    deadly word, using the inverted index of the book
    return:
        data frame of total number of occurance of each name in all books in the same sentece with a deadly word
    """
    deadly_dict = create_deadly_dict(books_inverted_index)
    match_count = []
    match_list = []
    for name in names_list:
        retrieved_docs = books_inverted_index.search(name, intersect_docs=True)
        # get the total count  of the name
        if len(retrieved_docs) == 0:
            match_count.append(0)

        else:
            # replace this with a dict of unified retrieval docs
            doc = retrieved_docs[0]
            match_dict = {}

            # find the matches between retrieved docs and the deadly words
            for book_num in range(5):
                if doc.get(book_num) is not None and deadly_dict.get(book_num) is not None:
                    name_word_list = [set(doc.get(book_num)), set(deadly_dict.get(book_num))]
                    occurance_set = set.intersection(*name_word_list)

                    if len(occurance_set) != 0:
                        match_dict[book_num] = occurance_set
                
                # else:
                #     match_dict[book_num] = {}
            
            match_list.append(match_dict)
            # add count per book
            count_per_book = [len(match_dict.get(i)) for i in range(5)
                            if match_dict.get(i) is not None]
            total_count = sum(count_per_book)
            match_count.append(total_count)

    # print(match_list)
    # print(match_count)
    return pd.DataFrame(match_count, columns=['deadly_word'])


def tf_idf_proximity_deadly_word(names_list, books_inverted_index):
    tf_idf_list = []
    i = 0
    print("creating tf-idf feature")
    for name in tqdm(names_list):
        retrieved_docs = books_inverted_index.search(name) # , intersect_docs=True
        if len(retrieved_docs) == 0:
            
            # uncomment this to see which names are not in the books
            # print(name, "is not found in the books")
            tf_idf_list.append(0)

        else:
            name_query = ' '.join(deadly_list) + ' ' + name
            docs = retrieved_docs[0]
            sentence_list = []
            for book_num in docs:
                for sentence in docs[book_num]:
                    sentence_list.append(books_inverted_index.\
                        sentences[book_num][sentence])

            big_sentence = ' '.join(sentence_list)
            deadly_index = Index()
            # deadly_index.add_sentence("empty")
            deadly_index.add_sentence(big_sentence)
            deadly_index.add_sentence("empty")
            tf_idf = deadly_index.tf_idf(0, name_query)
            tf_idf_list.append(tf_idf[0])

    return pd.DataFrame(tf_idf_list, columns=['tf_idf'])


def last_n_mentions(books_inverted_index, retrieved_docs, n):
    """ Find the last n times that a character name is mentioned in the book
    given the retrieved documents.

    Keywrod arguments:
        retrieved_docs: the dictionary of book-sentence shows where a
                        name appeard
        n: number of sentences we want to look at from the end of dictionary
    
    Returns:
        dict: books and sentences
    """
    count = 0
    last_n_sentence = ""
    # if mention count is less than n
    # create a reverse list out of the dict to find the latest mentions
    reverse_book_list = sorted(list(retrieved_docs), reverse=True)
    for book in reverse_book_list:
        for sentence in sorted(list(retrieved_docs[book]), reverse=True):
            last_n_sentence = " ".join([
                last_n_sentence,
                books_inverted_index.sentences[book][sentence].lower()
                ])
            count += 1
            if count >= n:
                return last_n_sentence

    # if mention count is more than or equal to n
    if count < n:
        return last_n_sentence

def count_deadly_words(sentence):
    """ Count the number of deadly words in the sentence."""
    count = 0
    for deadly_word in deadly_list:
        cur_count = sentence.count(deadly_word)
        if cur_count > 0:
            count += cur_count
    return count

def count_deadly_last_n_mentions(name_list, books_inverted_index, n):
    """ Create a feature of deadly words count in the last n sentences.
    
    Arguments:
        name_list {list} -- name of characters
        books_inverted_index {inverted_index} -- index for fast search
        n {int} -- last n sentence
    
    Returns:
        data_frame -- a column that can be added to the features matrix
    """
    count_list = []
    density_list = []
    for name in name_list:
        retrieved_docs = books_inverted_index.search(name, intersect_docs=True)[0]
        sentence = last_n_mentions(books_inverted_index, retrieved_docs, n)
        if len(sentence.split()) != 0:
            count = count_deadly_words(sentence)
            count_list.append(count)
            density_list.append(count / (len(sentence.split())))
        else:
            count_list.append(0)
            density_list.append(0)
    
    deadly_data = {'deadly_last_'+str(n)+'_mention': count_list,
                   'deadly_last_'+str(n)+'density': density_list
                  }
    return pd.DataFrame(deadly_data)

def density_deadly_last_n_mention(name_list, books_inverted_index, n):
    """ Create density of deadly words occurance around the last n mentions
        of each characters.

    Arguments:
        name_list {list} -- name of characters
        books_inverted_index {inverted_index} -- books' inverted index
        n {int} -- last n sentences

    Returns:
        data_frame -- a column that can be added to the features matrix
    """
    



def save_index(index):
    """ Save the inverted index as a pickle object."""    
    file = open('../data/inverted_index', 'wb')
    pickle.dump(index, file)
    file.close()


def load_index():
    """ Load the inverted index as a pickle object into its variable."""  
    try:
        file = open('../data/inverted_index', 'rb')
    except IOError:
        print('The inverted index file does not exist! Please run using index_books flag')
        sys.exit()
    return pickle.load(file)


def main(argv):
    # fake_scores = fake_score_generator(death_dataset)
    # death_dataset = add_feature(death_dataset, fake_scores)
    # names = {}
    # names_dict = create_name_dict(names_list)
    # filename = 'death'
    # names_list = get_file_names(filename)


    count_flag = False
    gender_flag = False
    proximity_flag = False
    tf_idf_flag = False
    graph_flag = False
    run_file_flag = False
    ablation_flag = False
    best_subset_flag = False

    books_inverted_index = Index()
    # do a fresh indexing and save
    # books_inverted_index.add_all_books()
    # save_index(books_inverted_index)
    # or load from previous indexing

    if len(sys.argv) > 1:

        for arg in sys.argv[1:]:
            if arg == 'index_books':
                # do a fresh indexing and save
                books_inverted_index.add_all_books()

                save_index(books_inverted_index)

            elif arg == 'load_books':
                print("loading books directly as inverted_index object into the program")
                books_inverted_index = load_index()

            elif arg == 'count_features':
                count_flag = True
            
            elif arg == 'gender_feature':
                gender_flag = True

            elif arg == 'proximity_feature':
                proximity_flag = True

            elif arg == 'tf_idf':
                tf_idf_flag = True

            elif arg == 'ablation':
                ablation_flag = True

            elif arg == 'best_subset':
                best_subset_flag = True

            elif arg == 'graph_feature':
                graph_flag = True
                graph = Graph()

            elif arg == 'all_features':
                count_flag = True
                gender_flag = True
                proximity_flag = True
                tf_idf_flag = True
                graph_flag = True
                graph = Graph()

            elif arg == 'run_file':
                run_file_flag = True

            elif arg == 'quick':
                books_inverted_index = load_index()
                count_flag = True
                gender_flag = True
                proximity_flag = True
                tf_idf_flag = False
                graph_flag = False
                graph = Graph()


            else:
                sys.exit("Wrong usage!")

    else:
        books_inverted_index = load_index()
        count_flag = True
        gender_flag = True
        proximity_flag = True
        tf_idf_flag = True
        graph_flag = True
        graph = Graph()

    classifier = Classifier()
    classifier.read_separate_train_test_files(evaluate=True)
    # classifier.split_data()


    # reading names for training and test sets
    training_names = classifier.get_names(training=True)
    test_names = classifier.get_names(test=True)

    # creating features for the training set
    features_index, training_features = create_features(training_names, 
                            books_inverted_index, graph, count_flag, gender_flag,
                            proximity_flag, tf_idf_flag, graph_flag)
    # creating features for the test set
    features_index, test_features = create_features(test_names, 
                            books_inverted_index, graph, count_flag, gender_flag,
                            proximity_flag, tf_idf_flag, graph_flag)

    classifier.set_features(training_features, test_features)
    classifier.save_features()

    y_pred_log = classifier.logistic_regression()
    # classifier.svc_polynomial()
    # classifier.svc_guassian_kernel()
    y_pred_svc = classifier.svc_sigmoid()
    y_pred_dt = classifier.decision_tree()
    y_pred_knn = classifier.k_nearest_neighbors()
    y_pred_nb = classifier.naive_base()


    # create the run file out of the knn's results
    if run_file_flag == True:
        classifier.make_new_run_file(y_pred_dt, 'dt')
        classifier.make_new_run_file(y_pred_log, 'logit')
        classifier.make_new_run_file(y_pred_svc, 'svc')
        classifier.make_new_run_file(y_pred_knn, 'knn')
        classifier.make_new_run_file(y_pred_nb, 'naive')

    # classifier.feature_selection()

    classifier.plot_f1_scores(classifier.method_name, 
                                classifier.f_scores, 
                                plot_title='Death Prediction', 
                                file_name='f1_scores')


    y_pred_list = [y_pred_log, y_pred_svc, y_pred_dt, y_pred_knn, y_pred_nb]
    
    classifier.plot_with_error_bars('death', y_pred_list, 
                                    classifier.method_name, 'Death Prediction', 
                                    'death_fscore_error')
    


    if gender_flag:
        gender_training_features = training_features[2]
        gender_test_features = test_features[2]
        classifier.evaluate_gender_prediction(gender_training_features, 
                                            gender_test_features,
                                            print_flag=True)

    if ablation_flag:
        ablation_test(classifier, features_index, training_features,
                        test_features)

    if best_subset_flag:
        best_subset_selection(classifier, training_features, test_features)

    # debugging lines
    # proximity_deadly_words(training_names, books_inverted_index)
    # tf_idf_proximity_deadly_word(training_names, books_inverted_index)
    # books_inverted_index.add_book('001ssb.txt')
    # books_inverted_index.gender_predictor('jon snow')

    # books_inverted_index.print_search_results("jon snow")
    # books_inverted_index.print_search_results("viserys targaryen")


        
if __name__ == "__main__":
    main(sys.argv[1:])
