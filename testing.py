'''
Kiho Noh and Sophia Wei
CSE163 Final Project: Online Gaming
This program tests the data processing function, all the machine learning
functions and all the data visualization functions utilized in this project.
'''


import pandas as pd
import data_processing
import machine_learning
import data_visualizations


def test_data_processing(df):
    '''
    This function tests the data processing method and returns the end
    dataframe after the processing.
    '''
    print(f'Length of Dataset before Filtering: {len(df)}')
    data = data_processing.process_data(df)
    print(f'Filtered Dataset Length: {len(data)}')

    return data


def test_machine_learning(df):
    '''
    This function tests each of the machine learning methods used to answer
    the 3 research questions and displays the error for each ML model's
    predictions on the data.
    '''
    gade_accuracy = machine_learning.predict_GADE(df)
    print(f'Accuracy in Predicting an individual\'s GADE: {gade_accuracy}')

    hours_error = machine_learning.predict_hours(df)
    print(f'Error in Predicting Online Gaming Hours: {hours_error}')

    error, swlhours_error, gadspin_error = machine_learning.ml_narcissism(df)
    print('Error in Narcissism prediction with Hours and all',
          f'Mental Health columns input: {error}')
    print('Error in Narcissism prediction with GAD and SPIN scores input:',
          f'{swlhours_error}')
    print('Error in Narcissism prediction with SWL score inputs:',
          f'{gadspin_error}')


def test_data_vis(df):
    '''
    This function tests each of the data visualization methods used to give
    more information and insight for the 3 research questions.
    '''
    data_visualizations.avg_GAD_over_20(df)
    data_visualizations.avg_GAD_under_20(df)
    data_visualizations.avg_hours_work(df)
    data_visualizations.hours_game_age(df)
    data_visualizations.narcissism_gaming_hours(df)
    data_visualizations.narcissism_over_20_hours(df)
    data_visualizations.narcissism_mental_health(df, 'GAD_T')
    data_visualizations.narcissism_mental_health(df, 'SPIN_T')
    data_visualizations.narcissism_mental_health(df, 'SWL_T')
    print('Data Visualizations for Research Questions Outputted!')


def main():
    df = pd.read_csv("GamingStudy_data.csv")
    data = test_data_processing(df)
    test_machine_learning(data)
    test_data_vis(data)


if __name__ == '__main__':
    main()
