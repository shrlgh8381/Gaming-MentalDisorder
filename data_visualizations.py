'''
Kiho Noh and Sophia Wei
CSE163 Final Project: Online Gaming
This program uses the pandas library and the matplotlib and seaborn libraries
to implement the graphing functions in order to better display the data and
to illustrate any patterns and/or correlations between variables in the data.
This is to bring better insight and extra information to help answer the 3
research questions.
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_processing

sns.set()


def avg_GAD_over_20(data):
    """
    Saves a line graph that plots the average weekly
    gaming hours ( >= 20) on the x-axis and the average total GAD
    score on the y-axis to visualize the difference in
    the average total GAD score of gamers who spend more than 20 hours
    a week on gaming.
    """
    data = data[["GADE", "GAD_T", "Hours"]]

    hours_over_20 = data[(data["Hours"] >= 20)]

    hours_over_20 = hours_over_20.groupby("Hours",
                                          as_index=False)["GAD_T"].mean()

    sns.lineplot(data=hours_over_20, x="Hours", y="GAD_T")

    plt.xlabel("Average Gaming Hours")
    plt.ylabel("Average GAD Score")
    plt.title("GAD Scores of Individuals Who Game More\
    than 20 Hours a Week")

    plt.savefig("hours_over_20.png")


def avg_GAD_under_20(data):
    """
    Saves a line graph that plots the average weekly
    gaming hours ( < 20) on the x-axis and the average total GAD
    score on the y-axis to visualize the difference in the average
    total GAD score of a gamer who spend less than 20 hours a week on gaming.
    """
    data = data[["GADE", "GAD_T", "Hours"]]

    hours_under_20 = data[(data["Hours"] < 20)]

    hours_under_20 = hours_under_20.groupby("Hours",
                                            as_index=False)["GAD_T"].mean()

    sns.relplot(data=hours_under_20, x="Hours", y="GAD_T", kind='line')

    plt.xlabel("Average Gaming Hours")
    plt.ylabel("Average GAD Score")
    plt.title("GAD Scores of Individuals Who Game Under 20hrs/Week")

    plt.savefig("hours_under_20.png", bbox_inches='tight')


def avg_hours_work(data):
    """
    Saves a bar graph that plots the current occupation of individuals
    on the x-axis and the average weekly gaming hours on the y-axis to
    visualize the relationship between the average weekly gaming hours
    and the current occupation status (unemployed/between jobs, student
    at college, employed) of gamers.
    """
    unemployed = data["Work"] == "Unemployed / between jobs"
    at_college = data["Work"] == "Student at college / university"
    employed = data["Work"] == "Employed"
    data = data[(at_college | unemployed | employed)]

    data = data.groupby("Work", as_index=False)["Hours"].mean()

    graph = sns.barplot(data=data, x="Work", y="Hours")

    plt.xlabel("Current Occupation")
    plt.ylabel("Average Gaming Hours")
    plt.title("Average Weekly Gaming hours of College students\
    and Unemployed Individuals")
    graph.set_xticklabels(graph.get_xticklabels(), fontsize=8)
    plt.savefig("work-gaming.png", bbox_inches='tight')


def hours_game_age(data):
    """
    Saves a bar graph that plots the age of individuals
    on the x-axis and the average weekly gaming hours on the y-axis to
    visualize the relationship between the average weekly gaming hours
    and the age of the gamers.
    """
    data = data.groupby("Age", as_index=False)["Hours"].mean()

    graph = sns.barplot(data=data, x="Age", y="Hours")

    plt.xlabel("Age")
    plt.ylabel("Average Gaming Hours")
    plt.title("Average Weekly Gaming hours of Different Age Groups")
    graph.set_xticklabels(graph.get_xticklabels(), fontsize=9)
    plt.savefig("age-gaming.png", bbox_inches='tight')


def narcissism_gaming_hours(df):
    '''
    This function takes in a dataframe and plots a scatterplot visualization
    between online gaming hours and narcissistic tendencies and then saves
    this graph.
    '''
    filtered_df = df[['Hours', 'Narcissism']]

    plt.figure()
    sns.relplot(x='Hours', y='Narcissism', hue='Narcissism',
                style='Narcissism',
                data=filtered_df).set(yticks=[1, 2, 3, 4, 5])

    plt.title('Narcissism & Weekly Online Gaming')
    plt.xlabel('Weekly Gaming Hours')
    plt.ylabel('Narcissistic Tendencies')
    plt.savefig('onlinegamingnarcissism.png', bbox_inches='tight')


def narcissism_over_20_hours(df):
    '''
    This function takes in a dataframe and plots a regression visualization
    between excessive online gaming hours (over 20 weekly hours) and
    narcissistic tendencies and then saves this graph.
    '''
    filtered_df = df[['Hours', 'Narcissism']]
    over_20_hours = df['Hours'] > 20
    filtered_df = filtered_df[over_20_hours]

    plt.figure()
    sns.set_theme(style="whitegrid")
    sns.regplot(x='Hours', y='Narcissism', color='red', fit_reg=True,
                ci=95, scatter_kws={'s': 5},
                data=filtered_df).set(xticks=[0, 25, 50, 75, 100,
                                              125, 150, 175])

    plt.title('Narcissism & Excessive Online Gaming')
    plt.xlabel('Weekly Gaming Hours')
    plt.ylabel('Narcissistic Tendencies')
    plt.savefig('excessivegamingnarcissism.png', bbox_inches='tight')


def narcissism_mental_health(df, mental_health):
    '''
    This function takes in a dataframe and makes various plots drawing
    box-plot and swarmplot visualizations to show the relationship between
    mental health disorders, like GAD and Social Phobia and Satisfaction
    with Life, and Narcissism. Then it saves 3 files, one for each graph.
    '''
    filtered_df = df[['Narcissism', 'GAD_T', 'SWL_T', 'SPIN_T']]
    high_narcissism = (filtered_df['Narcissism'] == 4) | \
                      (filtered_df['Narcissism'] == 5)
    factor = filtered_df[mental_health]
    filtered_df = filtered_df[high_narcissism & factor]

    plt.figure()
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='Narcissism', y=mental_health, hue='Narcissism',
                color='green', saturation=0.7, data=filtered_df)
    sns.swarmplot(x='Narcissism', y=mental_health, hue='Narcissism',
                  color='pink', size=1, data=filtered_df)

    plt.title(f'Narcissism & {mental_health}')
    if (mental_health == 'GAD_T'):
        mental_health = 'Generalized Anxiety Disorder'
    elif (mental_health == 'SWL_T'):
        mental_health = 'Satisfaction With Life'
    else:
        mental_health = 'Social Phobia Inventory Score'

    plt.xlabel('High Narcissistic Tendencies')
    plt.ylabel(f'{mental_health}')
    plt.savefig(f'{mental_health}.png', bbox_inches='tight')


def main():
    df = pd.read_csv("GamingStudy_data.csv")
    data = data_processing.process_data(df)
    avg_GAD_over_20(data)
    avg_GAD_under_20(data)
    avg_hours_work(data)
    hours_game_age(data)
    narcissism_gaming_hours(data)
    narcissism_over_20_hours(data)
    narcissism_mental_health(data, 'GAD_T')
    narcissism_mental_health(data, 'SPIN_T')
    narcissism_mental_health(data, 'SWL_T')


if __name__ == '__main__':
    main()
