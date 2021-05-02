'''
Kiho Noh and Sophia Wei
CSE163 Final Project: Online Gaming
This program uses the pandas library to implement the data processing method
in order to read through an Online Gaming dataset CSV file and parse down to
the columns of interest.
'''


def process_data(df):
    '''
    This function takes in a dataframe, filters it and returns a new
    dataframe with the columns of interest.
    '''
    filtered_df = df.loc[:, ["GADE", 'Hours', 'Narcissism', 'Age', 'Work',
                             'GAD_T', 'SWL_T', 'SPIN_T']]

    filtered_df = filtered_df[(filtered_df['Hours'] <= 168)]
    filtered_df = filtered_df.dropna()

    return filtered_df
