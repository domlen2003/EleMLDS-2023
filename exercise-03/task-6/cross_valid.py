import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    # loading the dataset
    titanic_df = pd.read_csv("titanic.csv")
    descriptive_features = ["Fare", "Age", "Sex", "Pclass"]
    target_features = ["Survived"]

    # in order for all cross validation sets to have the same size, we consider
    # 1/3 of the dataset for each one
    cross_valid_set1 = titanic_df.sample(frac=1/3, random_state=358605)
    cross_valid_set2 = titanic_df.drop(cross_valid_set1.index).sample(frac=1/2, random_state=358605)
    cross_valid_set3 = titanic_df.drop(cross_valid_set1.index).drop(cross_valid_set2.index)

    ###### Construct the three different train and test datasets here #######
    # uncomment the following four lines and complete them accordingly
    cv_train1 = pd.concat([cross_valid_set1, cross_valid_set2], axis=0)
    cv_test1 = cross_valid_set3
    cv_train2 = pd.concat([cross_valid_set1, cross_valid_set3], axis=0)
    cv_test2 = cross_valid_set2
    cv_train3 = pd.concat([cross_valid_set2, cross_valid_set3], axis=0)
    cv_test3 = cross_valid_set1

    ###### Split the datasets into descriptive and target features here ######
    # uncomment the following lines and complete them accordingly
    cv_train1_x = cv_train1[descriptive_features]
    cv_train1_y = cv_train1[target_features]
    cv_test1_x = cv_test1[descriptive_features]
    cv_test1_y = cv_test1[target_features]

    cv_train2_x = cv_train2[descriptive_features]
    cv_train2_y = cv_train2[target_features]
    cv_test2_x = cv_test2[target_features]
    cv_test2_y = cv_test2[target_features]

    cv_train3_x = cv_train3[descriptive_features]
    cv_train3_y = cv_train3[target_features]
    cv_test3_x = cv_test3[descriptive_features]
    cv_test3_y = cv_test3[target_features]


    ###### Training the models on each fold ##########
    # uncomment and complete the following lines accordingly
    log_reg1 = LogisticRegression()
    log_reg1.fit(cv_train1_x, cv_train1_y.values.ravel())
    log_reg2 = LogisticRegression()
    log_reg2.fit(cv_train2_x, cv_train2_y.values.ravel())
    log_reg3 = LogisticRegression()
    log_reg3.fit(cv_train3_x, cv_train3_y.values.ravel())

    dec_tree1 = DecisionTreeClassifier()
    dec_tree1.fit(cv_train1_x, cv_train1_y.values.ravel())
    dec_tree2 = DecisionTreeClassifier()
    dec_tree2.fit(cv_train2_x, cv_train2_y.values.ravel())
    dec_tree3 = DecisionTreeClassifier()
    dec_tree3.fit(cv_train3_x, cv_train3_y.values.ravel())


    """
    ###### Testing the models on each fold ##########
    # uncomment and complete the following lines accordingly
    # raise NotImplementedError("Complete the code here")
    log_reg1_pred = log_reg1.predict(cv_test1_x)
    log_reg1_conf_matrix = confusion_matrix(cv_test1_y, log_reg1_pred)
    #log_reg2_pred =
    #log_reg2_conf_matrix =
    #log_reg3_pred =
    #log_reg3_conf_matrix =

    dec_tree1_pred = dec_tree1.predict(cv_test1_x)
    dec_tree1_conf_matrix = confusion_matrix(cv_test1_y, dec_tree1_pred)
    #dec_tree2_pred =
    #dec_tree2_conf_matrix =
    #dec_tree3_pred =
    #dec_tree3_conf_matrix =


    def precision(cm):
        "Compute the precision of a model given its confusion matrix"
        raise NotImplementedError("Complete the code here")

    def recall(cm):
        "Compute the recall of a model given its confusion matrix"
        raise NotImplementedError("Complete the code here")

    def f1_score(cm):
        "Compute the f1-score of a model given its confusion matrix"
        raise NotImplementedError("Complete the code here")

    ###### Printing the results here ##########
    # Compute the average precision, recall and f1-score for each model and print it
    raise NotImplementedError("Complete the code here")
    
    """