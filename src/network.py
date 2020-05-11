import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class Graph(object):
    def __init__(self):
        self.graph = None
        self.training_names = self.get_names(training=True)
        self.test_names = self.get_names(test=True)
        self.names_list = self.combined_names()
        self.graph_created_flag = False
        self.features = None
        self.count_matrix = None
        

    def get_graph_features(self, target_names, inverted_index, method):
        """Creates the graph from the chosen method and creates the features
        matrix of that graph
        
        Arguments:
            target_names {list} -- list of names in the feature matrix; can be
                                    test_names or training_names
            inverted_index {inverted_index} -- inverted index of the document
            method {string} -- method to create the graph
        
        Returns:
            DataFrame -- data frame of features that is created using the method
        """        
        if not self.graph_created_flag:
            if method == 'characters_in_same_sentence':
                print("Creating the graph...")
                self.create_weighted_graph_chars_same_sent(inverted_index)
                print("Creating features from the graph...")
                self.create_graph_features()
                print("Creating graph features is done!")

            self.graph_created_flag = True
        
        features = self.features
        names_series = pd.Series(target_names)
        names_series = names_series.str.replace(r'\(|\)|\[|\]', '').str.lower()
        
        dict_feature_target = {feature : 
                [features[feature][name] for name in names_series] 
                                        for feature in features
                                        }
        return pd.DataFrame(dict_feature_target)


    def create_graph_kb(self, inverted_index):
        """Creates a graph of connected characters based on the wiki. Two characters are connected if they appeard in the same page in wiki.
    
        Arguments:
            name_list {list} -- list of all names in the data sets
            inverted_index {inverted_index} -- inverted index of wiki
        """    
        name_list = self.names_list
        G = nx.Graph()
        G.add_nodes_from(name_list)
        count_matrix = np.zeros([len(name_list), len(name_list)])
        # count_dict = {}
        remainder_names = list(name_list)
        retrieved_names = {}

        # creating a list of search results for names in the wiki 
        for name in name_list:
            retrieved_names[name] = inverted_index.search(name,
                                                        intersect_docs=True)

        for id0, name0 in enumerate(name_list):
            remainder_names.remove(name0)

            # [0] is for having just one document in inverted_index struct
            if len(retrieved_names[name0][0]) != 0:
                for id1, name1 in enumerate(remainder_names):
                    if len(retrieved_names[name0][0]) != 0:
                        # find all matches between document # in documents 
                        # retrieved by name0 and name1
                        for page_id in retrieved_names[name0][0]:
                            if retrieved_names[name1][0].get(page_id) is not None:
                                # add that darn id0 to compensate for the name
                                # removal
                                count_matrix[id0, id0 + id1] += 1
                                count_matrix[id0 + id1, id0] += 1
                                G.add_edge(name0, name1)
                                G.add_edge(name1, name0)

        # print(np.max(count_matrix), np.min(count_matrix))
        self.graph = G


    def create_weighted_graph_kb(self, inverted_index):
        """Creates a graph of connected characters based on the wiki. Two characters are connected if they appeard in the same page in wiki.
    
        Arguments:
            name_list {list} -- list of all names in the data sets
            inverted_index {inverted_index} -- inverted index of wiki
        """    
        name_list = self.names_list
        count_matrix = np.zeros([len(name_list), len(name_list)])
        remainder_names = list(name_list)
        retrieved_names = {}

        # creating a list of search results for names in the wiki 
        for name in name_list:
            retrieved_names[name] = inverted_index.search(name,
                                                        intersect_docs=True)

        for id0, name0 in enumerate(name_list):
            remainder_names.remove(name0)

            # [0] is for having just one document in inverted_index struct
            if len(retrieved_names[name0][0]) != 0:
                for id1, name1 in enumerate(remainder_names):
                    if len(retrieved_names[name0][0]) != 0:
                        # find all matches between document # in documents 
                        # retrieved by name0 and name1
                        for page_id in retrieved_names[name0][0]:
                            if retrieved_names[name1][0].get(page_id) is not None:
                                # add that darn id0 to compensate for the name
                                # removal
                                count_matrix[id0, id0 + id1] += 1
                                count_matrix[id0 + id1, id0] += 1

        remainder_names = list(name_list)
        graph = nx.Graph()
        normed_count = count_matrix / count_matrix.max()
        for id0, name0 in enumerate(name_list):
            remainder_names.remove(name0)
            for id1, name1 in enumerate(remainder_names):
                graph.add_edge(name0, name1, weight=normed_count[id0, id1])
        # graph.add_nodes_from(name_list)

        # print(np.max(count_matrix), np.min(count_matrix))
        self.graph = graph
        self.count_matrix = normed_count

    def create_graph_chars_same_sent(self, book_inverted_index):
        """Creates a graph of connected characters based on the text books.
        It's believed this is a more accurate representation of connection
        between entities (characters).
        
        Arguments:
            book_inverted_index {inverted_index} -- the inverted index of the
                                    text book
        
        Raises:
            NotImplementedError: [description]
        """        
        name_list = self.names_list
        G = nx.Graph()
        G.add_nodes_from(name_list)
        count_matrix = np.zeros([len(name_list), len(name_list)])
        # count_dict = {}
        remainder_names = list(name_list)
        retrieved_names = {}

        # creating a list of search results for names in the wiki 
        for name in name_list:
            retrieved_names[name] = book_inverted_index.search(name,
                                                        intersect_docs=True)

        for id0, name0 in enumerate(name_list):
            remainder_names.remove(name0)

            # [0] is for having just one document in inverted_index struct
            if len(retrieved_names[name0][0]) != 0:
                for id1, name1 in enumerate(remainder_names):
                    if len(retrieved_names[name0][0]) != 0:
                        # find all matches between document # in documents 
                        # retrieved by name0 and name1
                        for book in retrieved_names[name0][0]:
                            if retrieved_names[name1][0].get(book) is not None:
                                for sentence in retrieved_names[name0][0][book]:
                                    if retrieved_names[name1][0][book].get(sentence) is not None:
                                        # add that darn id0 to compensate for 
                                        # the name removal
                                        count_matrix[id0, id0 + id1] += 1
                                        count_matrix[id0 + id1, id0] += 1
                                        G.add_edge(name0, name1)
                                        G.add_edge(name1, name0)

        # print(np.max(count_matrix), np.min(count_matrix))
        self.graph = G


    def create_weighted_graph_chars_same_sent(self, book_inverted_index):
        """Creates a graph of connected characters based on the text books.
        It's believed this is a more accurate representation of connection
        between entities (characters).
        
        Arguments:
            book_inverted_index {inverted_index} -- the inverted index of the
                                    text book
        
        Raises:
            NotImplementedError: [description]
        """        
        name_list = self.names_list
        count_matrix = np.zeros([len(name_list), len(name_list)])
        # count_dict = {}
        remainder_names = list(name_list)
        retrieved_names = {}

        # creating a list of search results for names in the wiki 
        for name in name_list:
            retrieved_names[name] = book_inverted_index.search(name,
                                                        intersect_docs=True)

        for id0, name0 in enumerate(name_list):
            remainder_names.remove(name0)

            # [0] is for having just one document in inverted_index struct
            if len(retrieved_names[name0][0]) != 0:
                for id1, name1 in enumerate(remainder_names):
                    if len(retrieved_names[name0][0]) != 0:
                        # find all matches between document # in documents 
                        # retrieved by name0 and name1
                        for book in retrieved_names[name0][0]:
                            if retrieved_names[name1][0].get(book) is not None:
                                for sentence in retrieved_names[name0][0][book]:
                                    if retrieved_names[name1][0][book].get(sentence) is not None:
                                        # add that darn id0 to compensate for 
                                        # the name removal
                                        count_matrix[id0, id0 + id1] += 1
                                        count_matrix[id0 + id1, id0] += 1

        # print(np.max(count_matrix), np.min(count_matrix))

        remainder_names = list(name_list)
        graph = nx.Graph()
        normed_count = count_matrix / count_matrix.max()
        for id0, name0 in enumerate(name_list):
            remainder_names.remove(name0)
            for id1, name1 in enumerate(remainder_names):
                graph.add_edge(name0, name1, weight=normed_count[id0, id1])

        self.graph = graph
        self.count_matrix = normed_count


    def create_graph_features(self):
        graph = self.graph
        deg_centrality = nx.degree_centrality(graph) 
        # print(deg_centrality)

        # in_deg_centrality = nx.in_degree_centrality(graph) 
        # out_deg_centrality = nx.out_degree_centrality(graph) 

        close_centrality = nx.closeness_centrality(graph) 
        # print(close_centrality) 

        bet_centrality = nx.betweenness_centrality(graph, 
                                                normalized = True,  
                                                endpoints = False) 
        # print(bet_centrality) 

        pr = nx.pagerank(graph, alpha = 0.8)
        # print(pr)

        features_dict = {'deg_center': deg_centrality,
            'between_center': bet_centrality,
            'close_center': close_centrality,
            'page_rank': pr
            }

        self.features = features_dict

    def plot_graph(self):
        graph = self.graph
        # plt.subplot(121)
        plt.imshow(self.count_matrix)
        plt.show()
        options = {
            'node_color': 'black',
            'node_size': .01,
            'line_color': 'grey',
            'linewidths': 0,
            'width': 0.01,
        }

        # nx.draw(graph, with_labels=True)
        # nx.draw_circular(graph, with_labels=True)
        # plt.subplot(122)

        # nx.draw_circular(graph, **options)
        # plt.show()


    def get_names(self, training=False, test=False):
        file_name = None
        if training:
            file_name = "../data/deaths-train.csv"
        if test:
            file_name = "../data/deaths-test.csv"

        df = pd.read_csv(file_name)
        df['Name'] = df['Name'].str.replace(r'\(|\)|\[|\]', '')
        names_df = df['Name'].str.lower().copy()
        return names_df

    def combined_names(self):
        return pd.concat([self.training_names, self.test_names])
        
def load_book_index():
    """ Load the inverted index as a pickle object into its variable."""  
    import pickle
    import sys
    try:
        file = open('../data/inverted_index', 'rb')
    except IOError:
        print('The inverted index file does not exist! Please run using index_books flag')
        sys.exit()
    return pickle.load(file)

def main():
    book_index = load_book_index()
    graph = Graph()
    # test_feature = graph.get_graph_features(graph.test_names, 
    #                     book_index, method='characters_in_same_sentence')

    # training_feature = graph.get_graph_features(graph.training_names, 
    #                     book_index, method='characters_in_same_sentence')
                    
    # print(test_feature.shape)
    # print(training_feature.shape)

    graph.create_weighted_graph_chars_same_sent(book_index)
    graph.plot_graph()


    
if __name__ == "__main__":
    main()

