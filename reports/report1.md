# Report for Submission 1 - Team 4

Many different classifiers have been tried on the `character-death.csv` file, and the best one up until now has been decision tree. This is somewhat expected, when many variables in the dataset are naturally categorical (that are sometimes being represented by a boolean value). In this situations, a classifier that is more able to distinguish between classes is more likely to do better.

Also, since we are detaching ourselves from the nicely structured csv file, we have to rely more on the data that is coming from either wiki or the books.

For this submission, I tried to extract some features from the book, and add them to the dataset. Features I have extracted from the book is the number of times each character appears in each book, and in total. These will add 6 more columns into our dataset.

Then I tried to run the dataset with new features on the classifier. Here is a comparison between these two runs:

![with count features](../files/f1_scores_w_cpb.png)

![without count features](../files/f1_scores_wo_cpb.png)

Another idea that I have tried was detecting the gender of a character, only using the surrounding sentence. The way I have implemented this was to search for a character's name, obtain the sentences the name occurred in, and search for different pronouns. My idea was if we do that for a limited number of occurrences for each character (for instance for only 20 times of occurrences of the character in the text), we will be able to identify its gender. So far, this is not working well, and I had to take it out of features. But, I am hoping that it's just an implementation issue, and it gets fixed with some tweaks.

In order to obtain a faster search result, I manage to implement an inverse indexing data structure. This will be very useful later on when I am working on knowledge bases and the graph representations of characters, since they are very easy to work with, and it's super fast!