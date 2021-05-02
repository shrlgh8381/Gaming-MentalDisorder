'''
Kiho Noh and Sophia Wei
CSE163 Final Project: Online Gaming
This program uses the pandas library and the sci-kit learn library in order
to implement machine learning functions and algorithms to find insights on the
data. This is to answer 3 complex research questions on online gaming habits
and patterns and how these affect and is affected by mental health disorders.
'''


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import data_processing


def predict_GADE(data):
    """
    For the training and testing, total GAD score and weekly gaming hours were
    used as features and GADE was used as labels. This function returns the
    model's accuracy.
    """
    data = data[["GADE", "GAD_T", "Hours"]]

    features = data.loc[:, data.columns != "GADE"]

    labels = data["GADE"]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model = DecisionTreeClassifier()

    model.fit(features_train, labels_train)

    # Test Accuracy
    test_predictions = model.predict(features_test)
    gade_accuracy = accuracy_score(labels_test, test_predictions)

    return gade_accuracy


def predict_hours(data):
    """
    For the training and testing, age of individuals and current work
    occupation status of the gamer (employed, unemployed/between jobs,
    student) were used as features and weekly gaming hours was used as labels.
    For the current work occupation, dummy variables are used to transform
    the categorical features into numerical values. This function returns the
    error of the model.
    """
    data = data[["Age", "Work", "Hours"]]

    features = data.loc[:, data.columns != "Hours"]
    features = pd.get_dummies(features)

    labels = data["Hours"]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)

    test_predictions = model.predict(features_test)
    test_MSE = mean_squared_error(labels_test, test_predictions)

    return test_MSE


def ml_narcissism(df):
    '''
    This function takes in a dataframe and uses machine learning on data of
    interest to find predictions and insights. Online gaming hours, gad, swl
    and spin scores were used as features and narcissism as labels. The
    function returns the errors for each of the 3 models created.
    '''
    filtered_df = df[['Hours', 'Narcissism', 'GAD_T', 'SWL_T', 'SPIN_T']]
    over_20_hours = filtered_df['Hours'] > 20
    filtered_df = filtered_df[over_20_hours]

    # Features: Hours, SWL, SPIN and GAD;
    # Weekly hours of online gaming & Total SWL score;
    # Total GAD score & Total SPIN score
    # Label: degree exhibiting narcissistic tendencies
    features = filtered_df.loc[:, filtered_df.columns != 'Narcissism']
    swlhours_feature = filtered_df.loc[:, ['Hours', 'SWL_T']]
    gadspin_feature = filtered_df.loc[:, ['GAD_T', 'SPIN_T']]
    labels = filtered_df['Narcissism']

    # Split up the data into Training Set & Testing Set
    features_train1, features_test1, labels_train1, labels_test1 = \
        train_test_split(features, labels, test_size=0.2)

    features_train2, features_test2, labels_train2, labels_test2 = \
        train_test_split(swlhours_feature, labels, test_size=0.2)

    features_train3, features_test3, labels_train3, labels_test3 = \
        train_test_split(gadspin_feature, labels, test_size=0.2)

    # Fit the model onto the data
    model1 = DecisionTreeRegressor()
    model1.fit(features_train1, labels_train1)

    model2 = DecisionTreeRegressor()
    model2.fit(features_train2, labels_train2)

    model3 = DecisionTreeRegressor()
    model3.fit(features_train3, labels_train3)

    # Find Predictions and Compute error on Testing Set
    predictions = model1.predict(features_test1)
    error = mean_squared_error(labels_test1, predictions)

    swlhours_predictions = model2.predict(features_test2)
    swlhours_error = mean_squared_error(labels_test2, swlhours_predictions)

    gadspin_predictions = model3.predict(features_test3)
    gadspin_error = mean_squared_error(labels_test3, gadspin_predictions)

    return error, swlhours_error, gadspin_error


def main():
    df = pd.read_csv("GamingStudy_data.csv")
    data = data_processing.process_data(df)
    predict_GADE(data)
    predict_hours(data)
    ml_narcissism(data)


if __name__ == '__main__':
    main()
