import numpy as np
import pandas as pd
import re
import nltk
import sys
import matplotlib.pyplot as plt
import networkx as nx

import trec_car.read_data as read_data

from classifiers import Classifier
from inverted_index import Index

class KnowledgeBase(object):
    """This class is primarily created for implementing entity linking. But, it can be used for the wiki pages as well.
    
    Arguments:
        object {[type]} -- [description]
    """    
    def __init__(self):
        self.pages = None
        self.pages_inverted_index = None
        self.page_name_dict = None
        self.training_names = self.get_names(training=True)
        self.test_names = self.get_names(test=True)
        self.names_list = None
        self.extract_info()

    def extract_info(self):
            # read cbor file
        f = open("../data/got.cbor", "rb")
        self.pages = list(read_data.iter_pages(f))
        f.close()

        self.names_list = pd.concat([self.training_names, self.test_names])

        # training_names = self.training_names
        # test_names = self.test_names
        # name_list = self.names_list
        # name_list = ['jon snow', 'arya stark']
        # dictionary of page_names(page title) -> page_id
        page_name_dict = {}
        # Parsing the whole wiki as inverted index. Every sentense
        # is indexed with the page_id. We only have one wiki, hense, one
        # document exist in the inverted index.
        page_name_index = Index()
        # dictionary of redirect names, inlinks and disambiguition names 
        # in the wiki to the page_name
        name_dict_base = {}

        id = 0
        for page in self.pages:
            # print(id, page.page_name)
            my_page_name = (page.page_name
                .replace('(', '')
                .replace(')', '')
                .replace('[', '')
                .replace(']', '')
                .lower()
                )

            extra_info = ''
            page_name_dict[my_page_name] = id

            if name_dict_base.get(my_page_name) == None:
                name_dict_base[my_page_name] = [my_page_name]

            # else:
            #     name_dict_base[my_page_name].append(my_page_name)

            if page.page_meta.inlinkAnchors != None:
                for inlink in page.page_meta.inlinkAnchors:
                    name_dict_base[my_page_name].append(inlink[0])
                    extra_info = ' '.join([extra_info, inlink[0]])
                    if name_dict_base.get(inlink[0]) == None:
                        name_dict_base[inlink[0]] = [my_page_name]
                    else:
                        name_dict_base[inlink[0]].append(my_page_name)


            if page.page_meta.redirectNames != None:
                for redirect_name in page.page_meta.redirectNames:
                    name_dict_base[my_page_name].append(redirect_name)
                    extra_info = ' '.join([extra_info, redirect_name])
                    if name_dict_base.get(redirect_name) == None:
                        name_dict_base[redirect_name] = [my_page_name]
                    else:
                        name_dict_base[redirect_name].append(my_page_name)


            if page.page_meta.disambiguationNames != None:
                for dis_name in page.page_meta.disambiguationNames:
                    name_dict_base[my_page_name].append(dis_name)
                    extra_info = ' '.join([extra_info, dis_name])
                    if name_dict_base.get(dis_name) == None:
                        name_dict_base[dis_name] = [my_page_name]
                    else:
                        name_dict_base[dis_name].append(my_page_name)
                
            page_name_index.add_sentence(' '.join([my_page_name, extra_info]))
            # page_name_index.add_sentence(my_page_name)

            id += 1

        page_name_index.finish_adding_sentences()
        self.pages_inverted_index = page_name_index
        self.page_name_dict = page_name_dict


    def get_names(self, training=False, test=False):
        file_name = None
        if training:
            file_name = "../data/deaths-train.csv"
        if test:
            file_name = "../data/deaths-test.csv"

        df = pd.read_csv(file_name)
        # df['Name'] = df['Name'].str.replace('(', '').str.replace(')', '').str.replace('[', '').str.replace(']', '')
        df['Name'] = df['Name'].str.replace(r'\(|\)|\[|\]', '')
        names_df = df['Name'].str.lower().copy()
        # names_df = df['Name'].copy()
        # names_df = df['Name'].str.lower()
        # names = [re.sub("[()]", "", x) for x in names_df]

        # print(names_df.shape[0], "names in the dataset")
        # print(names_df)
        return names_df

# find characters' name with no mention count in the books

def find_match_name_in_kb(names_list, page_name_dict):
    """
    find names of the death data set in knowledge base
    """
    full_name_page_name_dict = {}
    for name in names_list:
        if page_name_dict.get(name) is not None:
            full_name_page_name_dict[name] = page_name_dict[name]

    print("In page_title:\t\t", len(full_name_page_name_dict))


def compare_kb_with_dict(names_list, page_name_dict, page_name_index):
    """
    find names of the death data set in knowledge base
    """
    full_name_page_name_dict = {}
    for name in names_list:
        if page_name_dict.get(name) is not None:
            full_name_page_name_dict[name] = page_name_dict[name]
            retrieved_docs = page_name_index.search(name, intersect_docs=True)
            if len(retrieved_docs) > 0:
                if len(retrieved_docs[0]) == 0:
                    print(name)

    print("In page_title:\t\t", len(full_name_page_name_dict))


def find_occurance_name_in_kb(name_list, page_name_index):
    """
    finding names in the knowledge base, containing the page titles, page_meta's inlinks, redirect names, and disabiguation names
    """
    name_dict_base = {}
    unmatched_names = []
    for fullname in name_list:
        # names = fullname.split()

        # get a dict of length 5, the intersection of occurance of all tokens in the query
        search_fullname = page_name_index.search(fullname, intersect_docs=True)

        # create a dictionary of names occured anywhere in the knowledge base, title, page_meta, etc
        if len(search_fullname) > 0:
            retrieved_docs = search_fullname[0]
            # print(fullname, retrieved_docs)
            # print("########", fullname.upper(), "#######")
            # page_name_index.print_search(retrieved_docs, 2)
            if len(retrieved_docs) > 0:
                name_dict_base[fullname] = retrieved_docs
            # if 
            else:
                unmatched_names.append(fullname)

        else:
            unmatched_names.append(fullname)
            
    print("IN KB:\t\t\t", len(name_dict_base)) 
    print("NOT in KB:\t\t", len(unmatched_names))
    print("total:\t\t\t", len(name_dict_base) + len(unmatched_names))
    # print(name_dict_base)


def stats_on_wiki(knowledge_base):

    print("\t=== Traning set ===")
    find_match_name_in_kb(knowledge_base.training_names, knowledge_base.page_name_dict)
    print("\t=== Test set ===")
    find_match_name_in_kb(knowledge_base.test_names, knowledge_base.page_name_dict)
    # find_match_name_in_kb(names_list, name_dict_base)
    print("\t=== Total ===")
    find_match_name_in_kb(knowledge_base.names_list, knowledge_base.page_name_dict)

    print("\t=== Traning set ===")
    find_occurance_name_in_kb(knowledge_base.training_names, knowledge_base.pages_inverted_index)
    print("\t=== Test set ===")
    find_occurance_name_in_kb(knowledge_base.test_names, knowledge_base.pages_inverted_index)
    print("\t=== Total ===")
    find_occurance_name_in_kb(knowledge_base.names_list, knowledge_base.pages_inverted_index)


def coreference_resolution(text):
    import os
    import stanza
    from stanza.server import CoreNLPClient
    import json

    os.environ["CORENLP_HOME"] = "/home/soheil/Downloads/corenlp"

    # set up the client
    # with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos','lemma','ner', 'parse','dcoref'], timeout=5000, memory='2G', output_format='json') as client: 
    # with CoreNLPClient(annotators=['pos','lemma','ner', 'parse','coref'], timeout=5000, memory='2G') as client: 

        # properties={'annotators': 'coref', 'coref.algorithm' : 'statistical'}

    client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'dcoref'], memory='2G', timeout=5000, output_format='json') # 'dcoref' to do multipass sieve 'tokenize', 'ssplit', , endpoint='http://localhost:9001'
    # for sieve
    # annotators = tokenize, ssplit, pos, lemma, ner, parse, dcoref

    # annotators needed for coreference resolution
    # annotators = pos, lemma, ner, parse    


    # print(client)
    # Start the background server and wait for some time
    client.start()

    # Print background processes and look for java
    # ps -o pid,cmd | grep java

    text = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."
    ann = client.annotate(text)

    # Shut down the background CoreNLP server
    client.stop()

    # print(ann['corefs'])
    for word in ann['corefs']:
        print(word['text'])

    # result = json.loads(ann)
    # num, mentions = result['corefs'].items()[0]
    # for mention in mentions:
    #     print(mention)



def coref_spacy(text):
    import spacy
    # import neuralcoref
    nlp = spacy.load('en_core_web_sm')
    # neuralcoref.add_to_pipe(nlp)

    # spacy.prefer_gpu()
    
    # doc = nlp(text)
    # resolved_text = doc._.coref_resolved
    # sentences = [sent.string.strip() for sent in nlp(resolved_text).sents]
    # output = [sent for sent in sentences if 'president' in 
    #         (' '.join([token.lemma_.lower() for token in nlp(sent)]))]
    # print('Fact count:', len(output))
    # for fact in range(len(output)):
    #     print(str(fact+1)+'.', output[fact])

    doc1 = nlp('My sister has a dog. She loves him.')
    print(doc1._.coref_clusters)

    doc2 = nlp('Angela lives in Boston. She is quite happy in that city.')
    for ent in doc2.ents:
        print(ent._.coref_cluster)


def stanford_openie():
    from openie import StanfordOpenIE

    with StanfordOpenIE() as client:
        text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
        print('Text: %s.' % text)
        for triple in client.annotate(text):
            print('|-', triple)

        graph_image = 'graph.png'
        client.generate_graphviz_graph(text, graph_image)
        print('Graph generated: %s.' % graph_image)

        # with open('corpus/pg6130.txt', 'r', encoding='utf8') as r:
        #     corpus = r.read().replace('\n', ' ').replace('\r', '')

        # triples_corpus = client.annotate(corpus[0:50000])
        # print('Corpus: %s [...].' % corpus[0:80])
        # print('Found %s triples in the corpus.' % len(triples_corpus))
        # for triple in triples_corpus[:3]:
        #     print('|-', triple)

def load_book_index():
    """ Load the inverted index as a pickle object into its variable."""  
    import pickle
    try:
        file = open('../data/inverted_index', 'rb')
    except IOError:
        print('The inverted index file does not exist! Please run using index_books flag')
        sys.exit()
    return pickle.load(file)

def concat_into_doc(retrieved_docs, inverted_index):
    document = ''
    for book in retrieved_docs:
        for sentence in retrieved_docs[book]:
            if len(inverted_index.sentences[book][sentence].split()) < 40:
                document = ' '.join([document, 
                                    inverted_index.sentences[book][sentence]])

    return document

def deadly_relations(names_list, books_inverted_index):
    """Figure out if a name in the name_list is in the same sentence as a
    deadly word, using the inverted index of the book
    return:
        data frame of total number of occurance of each name in all books in the same sentece with a deadly word
    """
    retrieved_death = books_inverted_index.search('die')[0]
    character_died_dict = {}

    for name in names_list:
        name_word_occurance = {}
        retrieved_docs = books_inverted_index.search(name, intersect_docs=True)

        # continue only if there is something in the retrieved docs
        if len(retrieved_docs[0]) > 0:
            doc = retrieved_docs[0]

            # find the matches between retrieved docs and the deadly word
            for book_num in doc:
                # for sentence in doc[book_num]:
                if retrieved_death.get(book_num) is not None:
                    name_word_list = [
                        set(doc[book_num]), 
                        set(retrieved_death[book_num])
                        ]
                    occurance_set = set.intersection(*name_word_list)

                    if len(occurance_set) > 0:
                        name_word_occurance[book_num] = {
                            sentence:doc[book_num][sentence] 
                            for sentence in occurance_set
                            }
                        character_died_dict[name] = name_word_occurance
            
    return character_died_dict




def main(argv):


    kb = KnowledgeBase()
    # stats_on_wiki(kb)


    # coref_spacy()
    book_index = load_book_index()
    # retrieved documents for die
    rd_die = book_index.search("die")
    print(book_index.get_counts_per_book(rd_die))
    # book_index.print_search(rd_die[0])
    text = (concat_into_doc(rd_die[0], book_index))

    rd_kill = book_index.search("kill")
    print(book_index.get_counts_per_book(rd_kill))

    # coreference_resolution(text[0 : 9999])
    coref_spacy(text)

    # Deadly relation doesn't work that well!
    # char_word_occurance = deadly_relations(kb.names_list, book_index)
    # print(len(char_word_occurance))
    # for chars in char_word_occurance:
    #     book_index.print_search(char_word_occurance[chars], 20)


    # stanford_openie()
    # compare_kb_with_dict(names_list, page_name_dict, pages_inverted_index)
    # print(page_name_dict.get("wyman manderly"))
    # print(pages_inverted_index.search("wyman manderly", intersect_docs=True))
    # print(page_name_dict.get("yohn royce"))
    # print(pages_inverted_index.search("yohn royce", intersect_docs=True))


    # tf-idf on sentences
    # body = ["This paper consists of survey of many papers", "another Document consist of strange examples to see if it works okay", "something else without the stuff"]
    # for sentence in body:
    #     page_name_index.add_sentence(sentence)
    # page_name_index.get_tf_idf(0, 'paper see Soheil')

if __name__ == "__main__":
    main(sys.argv[1:])


# alternative way to read the cbor file    
# with open("got.cbor", 'rb') as f:
#     # Pages is a list of dat.Page objects.
#     pages = list(read_data.iter_pages(f))
