# Team_4_cs953

In this project, we are trying to predict the odds of dying each character in the famous TV show, Game of Thrones. In order to do so, we use various sources of information, such as three datasets available in Kaggle.com, and the text in the books.

This is a Data Science project, and as in other data science settings, we use so many different tools from various places, such as; Machine Learning (ML), Knowledge Graphs and Information Retrieval. Because of that, tools from different platforms and programming languages have brought together to produce the most accurate resutls.

For the first stage of this project, I tried various ML classifiers in order to train a model for the characters, based on their various features, such as; gender, title, date of birth, mother, father, etc. 

The datasets are splited by their index. Odd indecies are used for testing and evens are used for training.

For evaluation, some well-known classification metrics such as *f1-score* and *confusion matrix* are used.


<!-- In order to do the IR side of the project, the tool `Lucene` is considered. This is a powerful tool to do tokenization and indexing of the big courpuses and make similarity scores based on the seached query. These information will later on be used to train a more accurate classifier. -->

# Requirements

* nltk 
* numpy 
* pandas 
* sklearn 
* seaborn 
* matplotlib
* tqdm

Please make sure to have `punkt` and `stopwords` submodules from nltk library installed. If you don't, you can install it using
```
python3
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
```
# Installation and Run
Please put all books and the `deaths-train.csv` and `death-test.csv` files in a folder called `data`. None of the features in the training and test sets are used for training the classifiers. All of the features are generated completely based on the books. If you manually download the books into the `data` folder, please make sure to run the `clean_books.sh` script in `scripts` folder. Otherwise, you can run the program using the following commands:
First, change the current directory to `src` and run the basic evaluation
```
cd src
python3 evaluation.py

```
This will take a bit of time to create all the features. Using this command, you load the pre-indexed books into the program as a pickle object, and generates all of the features. A quicker version of evaluation can be run using
```
python3 evaluation.py quick

```
Please use the quick flag if you want to see the results before 10 minutes!

To index the books and save them as a `pickle` object into the `data` directory (which is a time consuming job), you can run

```
python3 evaluation.py index_books all_features

```
If you only provide the books for the program, you have to use `index_book` flag. Otherwise, you can download the `pickle` object from the git repository and use it by issuing `load_book` flag, such as
```
python3 evaluation.py load_books all_features
```

The `run-file` will be created in the `output` directory, if you add a `run_file` flag to the command. You can then check the f1-score by running 

```
python3 evaluation.py load_books all_features run_file
cd ../scripts
./fscore.sh
```
All of the generated files (plots, run_file, csv of features, etc.) go under `output` directory.

You can also choose the features you want to include in the set of features using the following flags:
```
python3 evaluation.py load_books count_features gender_feature proximity_feature graph_feature tf_idf
```

In order to run ablation test and best subset selection and produce the respective plots, please use `ablation` and `best_subset` flags.