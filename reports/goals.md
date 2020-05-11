# Plan of Attach for Team 4
In this project we are predicting if a character of GoT dies or not. I will be applying some ML tools in order to solve the problem. In the first phase of the project, I have trained a few classifiers using some of the features we have access to via `death_characters.csv` file. In order to evaluate the performance of the classifiers, we will be using f1-score and the confusion matrix, which is already implemented in the code. 

For the next phases of the project, I would like to, firstly, explore the idea of feeding the classifier with the features that are not coming from the csv file, but rather, they are extracted from different resources, for instance natural language processing or network science.

In particular, I am interested into training a classifier that has the best performance with some features that we can extract from the corpus in some ways. Since we are splitting the problem into two sub-problems; a feature extractor and a classifier, we want to make sure that each part is working well. Therefore, at first, we try to find the most accurate classifier using the csv file's features.

Then, we try to extract the information from the text books and the wiki, in order to predict the features themselves, which then can be fed into the classifier to make prediction. Based on the feature that have been widely used by others, I believe extracting information from the text can be done through asking the following questions:

* which house does s/he belong to?
* which culture does s/he belong to?
* how old was s/he?
* who were his/her relatives/siblings/parents?
* are his/her relatives dead?
* what is the gender of X?
* ...


There are a lot of features that can be added to this model in order to enrich the set of features. One of them is the frequency of mention count of a character in the text. Another one is the part of the book that the character appears on. 

The second idea I would like to explore is to use a neural network straightly to on the text. This is a totally different approach, where instead of creating features and having a classifier to use those feature, we train a neural network to pick its own features. My initial guess is this method is going to work better than the first one, mainly because the first one is composed of two components that are not necessarily supposed to work well with each other. Having multiple components in a system adds a bit of flexibility and complexity to it, which is not always desirable.

Another idea to try is to create a knowledge graph of the characters, and apply some concepts of Network Science to those graphs to extract more features and characteristics from the information that is hidden beneath the text. This graphs can be created for different aspects of the story, for instance, we can have a graph of the house that characters belong to, and graph of siblings, and graph of friendship, etc. 


Here is a list of what I will be implementing:
- [ ] adding feature selection algorithm for picking the best features
- [ ] using nlp to find an answer to questions comes from features
- [ ] calculating the mention frequency of characters
- [ ] creating graphs out of the dataset, using either the book, or the wiki
- [ ] calculating a couple of graph measures
- [ ] implementing some graph walk algorithms, pagerank,...

