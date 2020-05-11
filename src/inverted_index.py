import numpy as np
import pandas as pd
import re
import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer
import sys
import pickle
from tqdm import tqdm


from classifiers import Classifier

class Index:
    """ 
    Inverted index 
    index: dictionary of tokens
    tokens: dictionary of books
    book: dictionay of sentence 
    sentence: list of location of occurance in a sentence
    """

    def __init__(self):
        self.index = {}
        self.sentences = {}
        self.sentence_id = 0
        self.retrieved_docs = []
        self.cur_query = None
        self.book_id = 0
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.done_adding = True
 

    def search_token(self, token, stopword_flag=True):
        """
        search for a token
        return:
            a dictionay of books 
        """

        if stopword_flag and token in self.stopwords:
            return
        word = EnglishStemmer().stem(token)

        return self.index.get(word)

        # for query_result in retrieved:


    def search(self, queries, intersect_docs=False, union_docs=False):
        """
        search a query that can contain multiple simple queries
        """
        if not self.done_adding:
            print("Adding sentences is not done yet!")
            sys.exit()

        self.cur_query = queries
        self.retrieved_docs = []
        tokens = [t.lower() for t in nltk.word_tokenize(queries)]
        for token in tokens:
            search_result = self.search_token(token)

            if search_result is not None:
                self.retrieved_docs.append(search_result)

        docs_count = len(self.retrieved_docs)

        # fixing the type of output
        if len(self.retrieved_docs) == 0:
            self.retrieved_docs = [{}]

        unified_doc = {}
        if union_docs:

            if docs_count <= 1:
                return self.retrieved_docs

            for doc in self.retrieved_docs:
                for book_num in range(5):
                    if doc.get(book_num) is not None:
                        for sentence in doc[book_num]:
                            # keys are sentence number
                            unified_doc[book_num][sentence] = [] # can be doc[book_num][key] but it shouldn't matter

            return [unified_doc]
        
        if intersect_docs:
            if docs_count < min(2, len(tokens)):
                return [{}]
            
            if docs_count == 1:
                return self.retrieved_docs

            for book_num in range(self.book_id):
                # set(doc.get(book_num)) for doc in self.retrieved_docs
                intersect_list = []
                intersect_set = {}
                for doc in self.retrieved_docs:
                    # the whole query haven't occured completely in this book
                    if doc.get(book_num) is None:
                        intersect_list = []
                        break
                    else:
                        intersect_list.append(set(doc.get(book_num)))

                if len(intersect_list) != 0:
                    intersect_set = set.intersection(*intersect_list)

                if len(intersect_set) != 0:
                    unified_doc[book_num] = {sentence: doc[book_num][sentence] for sentence in intersect_set} # dict.fromkeys(intersect_set, [])


            return [unified_doc]

        return self.retrieved_docs


    def add_sentence(self, sentence):
        """
        add a sentence of a book to the indexing
        """
        if self.done_adding:
            self.done_adding = False

        for location, token in enumerate([t.lower() for t in nltk.word_tokenize(sentence)]):
            if token in self.stopwords:
                continue

            token = EnglishStemmer().stem(token)
 
            # if self.sentence_id not in self.index[token]:
            if self.index.get(token) is None:
                self.index[token] = {}
            if self.index[token].get(self.book_id) is None:
                self.index[token][self.book_id] = {}
            if self.index[token][self.book_id].get(self.sentence_id) is None:
                self.index[token][self.book_id][self.sentence_id] = []
            self.index[token][self.book_id][self.sentence_id].append(location)

        if self.sentences.get(self.book_id) is None:
            self.sentences[self.book_id] = {}
        self.sentences[self.book_id][self.sentence_id] = sentence
        self.sentence_id += 1

    def finish_adding_sentences(self):
        self.book_id += 1
        self.done_adding = True

    def add_book(self, filename):
        """
        parse a book into senteces 
        """
        book_file = open("../data/" + filename)
        # book_file = open("../books/test.txt")
        book = book_file.read()
        book = book.replace("...", "")
        book = book.replace("\n", "")
        book = book.replace(". . .", "")
        sentences = nltk.sent_tokenize(book)
        print(len(sentences), "sentences in the book", filename)

        for sentence in tqdm(sentences):
            self.add_sentence(sentence)

        self.book_id += 1
        self.sentence_id = 0

    def add_all_books(self):
        """
        add all the books into the data set
        """
        filenames = ["001ssb.txt", "002ssb.txt", "003ssb.txt", "004ssb.txt", "005ssb.txt"] 
        for filename in filenames:
            self.add_book(filename)

        # make sure to lock, after adding all books
        self.done_adding = True

# inquery stop words
    def get_total_counts(self, retrieved_docs = None):
        """
        sum of elements in the list of query occurance over all books
        return:
            int total count
        """
        if retrieved_docs is None:
            retrieved_docs = self.retrieved_docs

        return sum(self.get_query_list_count(retrieved_docs))


    def get_counts_per_book(self, retrieved_docs):
        return self.get_query_list_count(retrieved_docs)


    def get_query_list_count(self, retrieved_docs):
        """
        the maximum occurance over all tokens in the query for each book
        return:
            a list of 5 counts
        """
        counts = []
        for ret_doc in retrieved_docs:
            counts.append(self.get_list_token_count(ret_doc))

        # print(counts)
        return list(np.max(counts, axis=0)) if len(counts) > 0 else [0, 0, 0, 0, 0]


    def get_list_sum_query_count(self, retrieved_docs = None):
        """
        a stat of the searched query (multiple tokens)
        return:
            list of list_count (for each query)
        """
        counts = []
        if retrieved_docs is None:
            retrieved_docs = self.retrieved_docs

        for ret_doc in retrieved_docs:
            counts.append(sum(self.get_list_token_count(ret_doc)))

        return counts


    def get_token_book_count(self, book_num, retrieved_docs):
        """
        count the number of occurance of a query in a given book
        using a given retrival result
        return:
            int count
        """
        count = 0
        if retrieved_docs != None:
            if retrieved_docs.get(book_num) != None:
                for sentence in retrieved_docs[book_num]:
                    count += len(retrieved_docs[book_num][sentence])
        return count


    def get_list_token_count(self, retrieved_docs):
        """
        counts the total number of occurance of a query in all books
        return:
            a 5 element list of counts
        """
        counts = [0, 0, 0, 0, 0]

        for i in range(5):
            counts[i] = self.get_token_book_count(i, retrieved_docs)
        
        return counts

    def tf(self, book_num, sentence_num, token):
        """
        tf: term frequency, count the number of occurance of a token in a given sentence
        in a given book, and then divide it by the length of the doc
        return:
            int count / length of the document
        """
        if self.search_token(token) != None:
            retrieved_docs_in_book = self.search_token(token).get(book_num)
            if retrieved_docs_in_book != None:
                token_locs = retrieved_docs_in_book.get(sentence_num)
                if token_locs != None:
                    return len(token_locs) / len(self.sentences.get(book_num).get(sentence_num).split())
        return 0

    def df(self, book_num, token):
        """
        df: document freq.: the number of times the term appeard in all documents
        if only the term appears once in a doc, we could count this as an occurance
        no need to know the number of times it appeard
        """
        retrieved_docs = self.search_token(token)
        if retrieved_docs != None:
            return(len(retrieved_docs.get(book_num)))
        return 0

    def idf(self, book_num, token):
        """
        inverse of document frequency
        """
        return np.log((self.sentence_id + 1) / (self.df(book_num, token) + 1))


    def tf_idf(self, book_num, query):
        tf_idf = []
        for sent_id in range(self.sentence_id):
            token_doc_tf = 0
            for token in query.split():
                # # calculate tf for a token
                # print("idf", self.idf(book_num, token))
                # print("tf", self.tf(book_num, sent_id, token))
                token_doc_tf += self.tf(book_num, sent_id, token) * self.idf(book_num, token)
            tf_idf.append(token_doc_tf)

        return tf_idf


    def print_search(self, retrieved_docs, print_count=9999, book_num=None):
        """Print the retrieved sentence using retrieved_doc object and to
        the number that is specified by print_count. You have this option
        to only print a specific book.
        
        Arguments:
            retrieved_docs {dict of dict} -- the result of a search, broken 
            into book number and sentence number
            print_count {int} -- number of sentence to be printed
        
        Keyword Arguments:
            book_num {int} -- You can choose to print only the retrieved
            docs in one specific book (default: {None})
        """        
        i = 0
        if book_num is None:
            for book in retrieved_docs:
                for sentence in retrieved_docs[book]:
                    if len(self.sentences[book][sentence].split()) < 40:
                        print("book:", book, "sentence:", sentence, self.sentences[book][sentence])
                    if i > print_count:
                        break
                    i += 1
        else:
            if retrieved_docs.get(book_num) is not None:
                for sentence in retrieved_docs[book_num]:
                        print("book:", book_num, "sentence:", sentence, self.sentences[book_num][sentence])
                        if i > print_count:
                            break
                        i += 1
