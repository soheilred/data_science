import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys

import matplotlib.font_manager
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report, accuracy_score, roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import train_test_split


class Classifier(object):
    def __init__(self):
        self.dataset = None
        self.x_train = None
        self.y_train = None
        self.x_test  = None
        self.y_test  = None
        self.names_df = None
        self.test_names = None
        self.train_names = None
        self.f_scores = []
        self.method_name = []
        self.filename = None


    def split_data(self):
        """
        spliting the training and test data
        """
        self.x_train = self.dataset.iloc[0::2].copy()
        self.x_test = self.dataset.iloc[1::2].copy()

        if self.filename == 'death':
            self.y_train = self.x_train['Died'].values
            self.y_test = self.x_test['Died'].values
            self.x_train.drop(['Died'], axis=1, inplace = True)
            self.x_test.drop(['Died'], axis=1, inplace = True)

        if self.filename == 'prediction':
            self.y_train = self.x_train['actual'].values
            self.y_test = self.x_test['actual'].values
            self.x_train.drop(['actual'], axis=1, inplace = True)
            self.x_test.drop(['actual'], axis=1, inplace = True)


    def read_data(self):
        """
        reads and cleans the data

        """
        death_var = 1
        alive_var = 0
        if self.filename == 'death':
            if self.filename == 'death':
                character_deaths = pd.read_csv("../data/character-deaths.csv")

            # names_df = character_deaths['Name'].str.lower()
            # names = [re.sub("[()]", "", x) for x in names_df]
            # character_deaths['Name'] = names
            character_deaths.loc[:, "Allegiances"] = pd.factorize(character_deaths.Allegiances)[0] # should I remove house from this column's elements?
            self.names_df = character_deaths["Name"].copy()

            character_deaths.loc[:,"Died"] = death_var
            character_deaths.loc[character_deaths["Death Year"].isnull(), "Died"] = alive_var

            character_deaths.fillna(value=-1, inplace=True)
            character_deaths.drop(["Name", "Death Year", "Book of Death", "Death Chapter"], axis=1, inplace = True)

            self.dataset = character_deaths


        if self.filename == 'prediction':
            character_predictions = pd.read_csv("../data/character-predictions.csv")
            
            death_preds = character_predictions.copy(deep = True)
            death_preds.loc[:, "culture"] = [get_cult(x) for x in death_preds.culture.fillna("")]
            death_preds.loc[:, "title"] = pd.factorize(death_preds.title)[0]
            death_preds.loc[:, "culture"] = pd.factorize(death_preds.culture)[0]
            death_preds.loc[:, "mother"] = pd.factorize(death_preds.mother)[0]
            death_preds.loc[:, "father"] = pd.factorize(death_preds.father)[0]
            death_preds.loc[:, "heir"] = pd.factorize(death_preds.heir)[0]
            death_preds.loc[:, "house"] = pd.factorize(death_preds.house)[0]
            death_preds.loc[:, "spouse"] = pd.factorize(death_preds.spouse)[0]

            death_preds.drop(["name", "alive", "pred", "plod", "isAlive", "dateOfBirth", "S.No", "DateoFdeath"], 1, inplace = True)
            death_preds.columns = map(lambda x: x.replace(".", "").replace("_", ""), death_preds.columns)
            death_preds.fillna(value = -1, inplace = True)

            # self.x_train.drop(["SNo", "actual", "DateoFdeath"], 1, inplace = True)
            # self.x_test.drop(["SNo", "actual", "DateoFdeath"], 1, inplace = True)

            self.dataset = death_preds
        
    def read_separate_train_test_files(self, evaluate=False, feature_from_dataset=False):
        """
        read the data from separate test and train files
        for training and test sets, we only read the names and the Died column
        and totally discard everything else
        """
        death_var = 1
        alive_var = 0
        train_set = pd.read_csv("../data/deaths-train.csv")
        train_set.loc[:, "Allegiances"] = pd.factorize(train_set.Allegiances)[0]
        train_set.loc[:,"Died"] = death_var
        train_set.loc[train_set["Death Year"].isnull(), "Died"] = alive_var
        train_set.fillna(value=-1, inplace=True)

        # save the names in the train set 
        self.train_names = train_set["Name"].values
        self.y_train = train_set['Died'].values

        # setting features and response for training set
        if feature_from_dataset == True:
            train_set.drop(["Name", "Death Year", "Book of Death", "Death Chapter"], axis=1, inplace = True)
            self.x_train = train_set.drop(['Died'], axis=1)

        # setting features and response for the test set (just the names and response, and no other columns)
        test_set = pd.read_csv("../data/deaths-test.csv")
        # should I clean the names? NO, because I need them unchanged for the output
        # .str.replace('(', '').str.replace(')', '').str.replace('[', '').str.replace(']', '').str.lower()

        # read and save the response value in the test set, if the flag has been set
        if evaluate == True:
            test_set.loc[:,"Died"] = death_var
            test_set.loc[test_set["Death Year"].isnull(), "Died"] = alive_var
            test_set.fillna(value=-1, inplace=True)
            self.y_test = test_set["Died"].values

        self.test_names = test_set['Name'].values


    def set_features(self, new_training_features, new_test_features):
        if len(new_training_features) != len(new_test_features):
            error_string = 'traing set has {} features'.\
                format(len(new_training_features))
            error_string += ', but test set has {} features'.\
                format(len(new_test_features))
            sys.exit(error_string)

        for feature in new_training_features:
            # add features to the training set
            if self.x_train is not None:
                self.x_train = self.add_feature(self.x_train, feature)
            else:
                self.x_train = feature.copy(deep=True)

        for feature in new_test_features:
            # add features to the test set
            if self.x_test is not None:
                self.x_test = self.add_feature(self.x_test, feature)
            else:
                self.x_test = feature.copy(deep=True)


    def add_feature(self, prev_features, new_feature):
        """
        add new columns into the dataset
        """
        assert(prev_features.shape[0] == new_feature.shape[0])
        return pd.concat([prev_features, new_feature], axis=1)


    def save_features(self):
        self.x_train.to_csv("../output/train_features.csv")
        self.x_test.to_csv("../output/test_features.csv")


    def get_names(self, training=False, test=False):
        if training == True:
            return self.train_names
        if test == True:
            return self.test_names

        return np.concatenate([self.train_names, self.test_names])



    def make_run_file(self, y_test):
        """
        create a run file using the prediction for the test set
        """
        # x_test = self.names_df.iloc[1::2].copy()
        test_names = self.test_names

        run_file = pd.DataFrame()
        run_file["is_relevent"] = y_test
        run_file["name"] = np.array(list(map(lambda x: x.replace(" ", "_"), test_names)))
        run_file["section"] = 0
        run_file["query"] = 0
        run_file["rank"] = 0
        run_file["score"] = 1
        run_file["teamname"] = "team_4"
        to_output = run_file[["query", "section", "name", "rank", "score", "teamname"]].loc[run_file["is_relevent"] == 1]
        to_output["rank"] = np.arange(len(to_output)) + 1
        to_output.to_csv('../output/predictions.run', header=None, index=None, sep=' ', mode='w')

    
    def make_new_run_file(self, y_test, file_name):
        """
        create a run file in the updated format 
        """
        # x_test = self.names_df.iloc[1::2].copy()
        test_names = self.test_names
        big_run_file = pd.DataFrame(columns=["is_relevent", "name", "section", "query", "rank", "score", "teamname"])
        for i in range(10):
            subset = pd.read_csv('../data/random-subsets/subset'+str(i)+'.txt', header=None)
            subset.columns = ['name']
            subset['name'] = subset['name'].str.replace(' ', '_')
            # test_subset_names.append(subset)
            # print(subset.head)
        # print(test_subset_names)

            run_file = pd.DataFrame()
            run_file["is_relevent"] = y_test
            run_file["name"] = np.array(list(map(lambda x: x.replace(" ", "_"), test_names)))
            run_file["section"] = 0
            run_file["query"] = i
            run_file["rank"] = 0
            run_file["score"] = 1
            run_file["teamname"] = "team_4"
            run_file = run_file[run_file["name"].isin(subset["name"])]
            big_run_file = big_run_file.append(run_file)

        to_output = big_run_file[
            ["query", "section", "name", "rank", "score", "teamname"]
            ].loc[big_run_file["is_relevent"] == 1]
        to_output["rank"] = np.arange(len(to_output)) + 1
        to_output.to_csv('../output/predictions_' + file_name + '.run',
                        header=None, index=None, sep=' ', mode='w')


    def plot_with_error_bars(self, target_type, y_pred_list,
                            method_name, plot_title, file_name):
        f1_score_list = np.zeros([10, len(y_pred_list)])
        test_names = self.test_names

        if target_type == 'death':
            y_test = self.y_test
        elif target_type == 'gender':
            test_set = pd.read_csv("../data/deaths-test.csv")
            test_set.fillna(value=0, inplace=True)
            y_test = test_set['Gender'].values
        else:
            sys.exit('Unspecified plot type')
        # for each name subset
        for i in range(10):
            subset = pd.read_csv('../data/random-subsets/subset'+str(i)+'.txt',
                                names=['name'], header=None)
            subset = subset['name'].values
            # subset.columns = ['name']
            # subset['name'] = subset['name'].str.replace(' ', '_')
            j = 0
            for y_pred in y_pred_list:
                target_indices = [name_ind for name_ind in range(len(test_names)) if test_names[name_ind] in subset]
                # print(target_indices)
                reduced_y_pred = y_pred[target_indices]
                reduced_y_test = y_test[target_indices]
                f1_score_list[i, j] = f1_score(reduced_y_test, reduced_y_pred)
                j += 1

        # print(f1_score_list)

        f1_scores = np.mean(f1_score_list, axis=0)
        errors = np.std(f1_score_list, axis=0)
        # print('sd for ' + plot_title + ' are', errors)
        # print(f1_scores.shape)
        # print(errors.shape)

        x_pos = np.arange(len(method_name))
        width = 0.25 

        if (len(f1_scores) != len(x_pos)):
            sys.exit('Plot with error bars has failed. Dimentions of' \
                    + 'f1-scores ({}) and methods ({}) are not matching'.\
                        format(len(f1_scores), len(x_pos)))

        fig, ax = plt.subplots()
        plt.grid(color='w', alpha=.35, linestyle='--')
        ax.patch.set_facecolor(color='gray')
        ax.patch.set_alpha(.35)
        rect = ax.bar(x_pos, f1_scores, align='center', width=width, alpha=0.9,
                        capsize=5, yerr=errors)
        # ax.set_xticks(y_pos, method_name)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_name)
        ax.set_ylim([0,1])
        plt.xlabel('Methods')
        plt.ylabel('F1-Score')
        ax.set_title(plot_title)
        # ax.legend(['f1-scores']) # we can add more measures to this plot

        def label_plot(rects):
            """
            Attach a text label above each bar displaying its height
            """
            # for rect in rects:
            #     height = rect.get_height()
            #     ax.text(rect.get_x() + rect.get_width()/2., height,
            #         '%.3f' % height,
            #         ha='center', va='bottom')
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.3f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(18, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10)

        label_plot(rect)
        # plt.legend()
        plt.tight_layout()

        plt.savefig('../output/' + file_name + '.png')
        # plt.show()
        plt.close()



    def plot_f1_scores(self, method_name, f1_scores, plot_title, file_name):
        matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') # '/usr/share/fonts/TTF/'
        plt.rcParams['font.family'] = "Comfortaa" # font.fantasy # sans-serif

        # plt.style.use('ggplot')
        y_pos = np.arange(len(method_name))
        width = 0.25 

        fig, ax = plt.subplots()
        plt.grid(color='w', alpha=.35, linestyle='--')
        ax.patch.set_facecolor(color='gray')
        ax.patch.set_alpha(.35)
        rect = ax.bar(y_pos, f1_scores, align='center', width=width, alpha=0.9)
        # ax.set_xticks(y_pos, method_name)
        ax.set_xticks(y_pos )
        ax.set_xticklabels(method_name)
        ax.set_ylim([0,1])
        plt.xlabel('Methods')
        plt.ylabel('F1-Score')
        ax.set_title(plot_title)
        # ax.legend(['f1-scores']) # we can add more measures to this plot

        def label_plot(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                    '%.3f' % height,
                    ha='center', va='bottom')

        label_plot(rect)
        # plt.legend()
        plt.tight_layout()

        plt.savefig('../output/' + file_name + '.png')
        # plt.show()
        plt.close()

    def get_f1scores(self):
        """Returns the currently available list of f1-scores, and reset the
        list for the next run.
        """        
        fscores = self.f_scores.copy()
        methods = self.method_name.copy()
        self.f_scores = []
        self.method_name = []
        return fscores, methods

    def add_count_features(self):
        """
        add total counts and per book counts into features 
        """
        total_counts_df = pd.read_csv("../output/total_count.csv")
        self.add_feature(total_counts_df['total_count'])

        count_per_book_df = pd.read_csv("../output/count_per_book.csv")
        self.add_feature(count_per_book_df[count_per_book_df.columns[1:6]])
        # print(count_per_book_df.columns[1:6])


    def logistic_regression(self):
        print("======== DEATH PREDICTION =========")
        logreg = LogisticRegression()
        logreg.fit(self.x_train, self.y_train)
        y_pred = logreg.predict(self.x_test)
        # print("Confusion Matrix:")
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('Logistic')
        print("="*9, "Logistic", "="*10)
        print(confusion_matrix(self.y_test,y_pred))
        print("f1-score\t: %.4f" % f_score)
        return y_pred


    def svc_linear(self):
        # for some reasons linear kernel takes too long to be trained
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test)
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('SVC linear')
        print("="*11, "SVC linear", "="*11)
        print(confusion_matrix(self.y_test,y_pred))
        print("f1-score\t: %.4f" % f_score)
        # print(classification_report(self.y_test,y_pred))
        return y_pred


    # polynomial kernel
    def svc_polynomial(self):
        svclassifier = SVC(kernel='poly', degree=8)
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test)
        # print(confusion_matrix(self.y_test, y_pred))
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('SVC poly')
        print("="*10, "SVC poly", "="*9)
        print(classification_report(self.y_test, y_pred))
        print("f1-score\t: %.4f" % f_score)
        return y_pred


        # Guassian kernel
    def svc_guassian_kernel(self):
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test)
        # print(classification_report(self.y_test, y_pred))
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('SVC guassian')
        print("="*7, "SVC guassian", "="*8)
        print(confusion_matrix(self.y_test, y_pred))
        print("f1-score for \t: %.4f" % f_score)
        return y_pred

    
    # sigmoid
    def svc_sigmoid(self):
        svclassifier = SVC(kernel='sigmoid')
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test)
        # print(classification_report(self.y_test, y_pred))
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('SVC sig')
        print("="*8, "SVC sig", "="*8)
        print(confusion_matrix(self.y_test, y_pred))
        print("f1-score\t: %.4f" % f_score)
        return y_pred


    # decision tree
    def decision_tree(self):
        classifier = DecisionTreeClassifier()
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('Decision Tree')
        print("="*7, "Decision Tree", "="*7)
        print(confusion_matrix(self.y_test, y_pred))
        print("f1-score\t: %.4f" % f_score)
        return y_pred
        # print(classification_report(self.y_test, y_pred))
        # [link](https://stackabuse.com/decision-trees-in-python-with-scikit-learn/)

    # KNN
    # TODO: similarity
    def k_nearest_neighbors(self):
        # for k in [2, 3, 5, 7, 9, 11, 13]:
        k = 9
        KNN_model = KNeighborsClassifier(n_neighbors=k)
        KNN_model.fit(self.x_train, self.y_train)
        y_pred = KNN_model.predict(self.x_test)
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('KNN k'+str(k))
        print("="*10, "KNN k =",k, "="*9)
        print(confusion_matrix(self.y_test, y_pred))
        print("f1-score\t: %.4f" % f_score)
        # precisioin =  precision_score(self.y_test, y_pred)
        # print("precision\t",precisioin)
        # recall = recall_score(self.y_test, y_pred)
        # print("recall\t", recall)
        # print(2 * precisioin * recall/(precisioin + recall))
        return y_pred
        # print(classification_report(self.y_test, y_pred))
        # [link](https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/)


    def naive_base(self):
        gnb = GaussianNB()
        gnb_fit = gnb.fit(self.x_train, self.y_train)
        y_pred = gnb_fit.predict(self.x_test)
        f_score = f1_score(self.y_test, y_pred)
        self.f_scores.append(f_score)
        self.method_name.append('Naive Bayes')
        print("="*10, "Naive Bayes", "="*9)
        print(confusion_matrix(self.y_test, y_pred))
        print("f1-score\t: %.4f" % f_score)
        return y_pred

    def feature_selection(self):
        # from sklearn.feature_selection import VarianceThreshold
        from sklearn.feature_selection import SelectKBest
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        from sklearn.feature_selection import chi2
        from sklearn.ensemble import ExtraTreesClassifier

        # if self.filename = 'death':
        #     y = self.dataset[['actual']]
        #     features = self.dataset.drop(['actual'], axis=1, inplace = False)
        # if self.filename = 'prediction':
        #     y = self.dataset[['actual']]
        #     features = self.dataset.drop(['actual'], axis=1, inplace = False)
        # else:
        #     return

        features = self.x_train
        y_train = self.y_train
        # sel = VarianceThreshold(threshold=(.8))
        # features = sel.fit_transform(features)


        # pca = PCA(n_components=3)
        # fit = pca.fit(features)

        # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, y_train)
        # model = SelectFromModel(lsvc, prefit=True)
        # X_new = model.transform(features)
        # print(X_new.columns)

        print(features.shape)
        sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
        sel.fit(features, y_train)
        selected_feat= features.columns[(sel.get_support())]
        # pd.series(sel.estimator_,feature_importances_,.ravel()).hist()

        print(selected_feat.shape)
        print(selected_feat)

        # Create a selector object that will use the random forest classifier to 
        # identify features that have an importance of more than 0.15
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(features, y_train)
        sfm = SelectFromModel(clf, prefit=True) # threshold=0.15
        features_new = sfm.transform(features)
        print(features_new.shape)


        # # Train the selector
        # sfm.fit(features, y_train)
        # X_important_train = sfm.transform(features)
        # X_important_test = sfm.transform(self.x_test)
        # # Create a new random forest classifier for the most important features
        # clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

        # # Train the new classifier on the new dataset containing the most important features
        # clf_important.fit(X_important_train, y_train)


    def evaluate_gender_prediction(self, training_gender_df, 
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

        y_pred_list = []

        print("======= GENDER PREDICTION =======")
        for column in test_gender_df.columns:
            y_pred = test_gender_df[column]
            y_pred_list.append(y_pred)
            # genders = test_gender_df['gender']
            sign_num = int((30 - len(column) - 2 ) / 2)
            f_score = f1_score(y_test, y_pred)
            f1_scores.append(f_score)
            if print_flag == True:
                print("="*sign_num, column, "="*sign_num)
                print(confusion_matrix(y_test, y_pred))
                print("f1-score\t: %.4f" % f_score)

        assert(len(f1_scores) == len(method_name))

        # Train a classifier using the features that were previously created 
        # from the text books. A few methods that were previously proven to
        # be working better with the data set are selected.

        # read the training set for obtaining the gender column
        train_set = pd.read_csv("../data/deaths-train.csv")
        train_set.fillna(value=0, inplace=True)
        y_train = train_set['Gender'].values

        cls_scores, cls_mtd_name, y_pred_cls = self.gender_classifier(
                                                    training_gender_df,
                                                    y_train, test_gender_df,
                                                    y_test, print_flag=True
                                                    )

        f1_scores = f1_scores + cls_scores
        method_name = method_name + cls_mtd_name
        y_pred_list = y_pred_list + y_pred_cls

        self.plot_f1_scores(method_name, 
                            f1_scores,
                            plot_title="Gender Prediction", 
                            file_name='gender_prediction')

        self.plot_with_error_bars('gender', y_pred_list, method_name,
                                    'Gender Prediction', 'gender_fscore_error')

        self.make_gender_run_file(y_pred)
        
    def gender_classifier(self, x_train, y_train, x_test, y_test, print_flag=False):
        """Trying three classifiers on the rule-based feature that is obtained
        for the gender. Features are:
            1. the number of pronoun in the sentence that a name occured and the next one determines the gender

            2. the closest pronoun determines the gender of a name
        return:
            a list of f1-scores
            a list of methods used
        """
        f1_scores = []
        method_name = []
        y_pred_list = []
        # train a logistic regression classifier
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        y_pred_logreg = logreg.predict(x_test)
        f_score = f1_score(y_test, y_pred_logreg)
        f1_scores.append(f_score)
        method_name.append('Logistic')
        if print_flag == True:
            print("="*10, "Logistic", "="*10)
            print(confusion_matrix(y_test,y_pred_logreg))
            print("f1-score\t: %.4f" % f_score)

        # train a SVC
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(x_train, y_train)
        y_pred_svc = svclassifier.predict(x_test)
        f_score = f1_score(y_test, y_pred_svc)
        f1_scores.append(f_score)
        method_name.append('SVC')
        if print_flag == True:
            print("="*13, "SVC", "="*13)
            print(confusion_matrix(y_test,y_pred_svc))
            print("f1-score\t: %.4f" % f_score)

        # train a decision tree
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        y_pred_dt = classifier.predict(x_test)
        f_score = f1_score(y_test, y_pred_dt)
        f1_scores.append(f_score)
        method_name.append('Decision Tree')
        if print_flag == True:
            print("="*8, "Decision Tree", "="*8)
            print(confusion_matrix(y_test,y_pred_dt))
            print("f1-score\t: %.4f" % f_score)

        y_pred_list.append(y_pred_logreg)
        y_pred_list.append(y_pred_svc)
        y_pred_list.append(y_pred_dt)

        return f1_scores, method_name, y_pred_list


    def make_gender_run_file(self, y_test):
        """
        create a run file in the updated format 
        """
        test_names = self.test_names
        big_run_file = pd.DataFrame(columns=['is_relevent', 'name', 'section',
                                             'query', 'rank', 'score', 
                                             'teamname'])
        qrel_file = pd.DataFrame(columns=['is_male', 'section', 'name', 'q_id'])
        gender_df = pd.read_csv('../data/deaths-test.csv')
        # gender_df['Name'] = gender_df['Name'].str.replace(r'\(|\)|\[|\]', '')
        gender_df['Name'] = gender_df['Name'].str.replace(r' ', '_')

        for i in range(10):
            subset = pd.read_csv('../data/random-subsets/subset'+str(i)+'.txt', header=None)
            subset.columns = ['name']
            subset['name'] = subset['name'].str.replace(' ', '_')
            # test_subset_names.append(subset)
            # print(subset.head)
        # print(test_subset_names)

            run_file = pd.DataFrame()
            run_file['is_relevent'] = y_test
            run_file['name'] = np.array(list(map(lambda x: x.replace(' ', '_'), test_names)))
            run_file['section'] = 0
            run_file['query'] = i
            run_file['rank'] = 0
            run_file['score'] = 1
            run_file['teamname'] = 'team_4'
            run_file = run_file[run_file['name'].isin(subset['name'])]
            big_run_file = big_run_file.append(run_file)


            qrel_chunk = pd.DataFrame()
            qrel_chunk['is_male'] = gender_df['Gender']
            qrel_chunk['section'] = 0
            qrel_chunk['name'] = np.array(list(map(lambda x: x.replace(' ', '_'), test_names)))
            qrel_chunk['q_id'] = i
            qrel_chunk = qrel_chunk[qrel_chunk['name'].isin(subset['name'])]
            qrel_file = qrel_file.append(qrel_chunk)


        to_output = big_run_file[
            ['query', 'section', 'name', 'rank', 'score', 'teamname']
            ].loc[big_run_file['is_relevent'] == 1]
        to_output['rank'] = np.arange(len(to_output)) + 1
        to_output.to_csv('../output/gender.run', header=None, index=None, sep=' ', mode='w')

        qrel_file = qrel_file.loc[qrel_file['is_male'] == 1]
        qrel_file.to_csv('../data/gender.qrels', header=None, index=None, sep=' ', mode='w', columns=['q_id', 'section', 'name', 'is_male'])



def get_cult(value):
    cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    }

    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()





