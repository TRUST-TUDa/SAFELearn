import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'


df = pd.read_csv("data/PPMI/PPMI_Curated_Data_Cut_Public_20230612_rev.csv")

columns_to_keep = ['PATNO', 'COHORT','age_at_visit', 'SEX', 'race', 'BMI', 'moca', 'fampd', 'bjlot', 'stai', 'scopa', 'CSFSAA', 'nfl_serum', 'asyn']
#df[columns_to_keep] = df[columns_to_keep].replace('.', np.nan)


average = []
std = []

for column in columns_to_keep:
    curr_sum = 0
    numbers_for_std = []
    for i in range(len(df[column])):
        if(df[column][i] == '.'):
            continue
        curr_sum += float(df[column][i])
        numbers_for_std.append(float(df[column][i]))
        
    average.append(curr_sum / len(numbers_for_std))
    #print(max(numbers_for_std))
    std.append(np.std(numbers_for_std, ddof=1))
    
for column in columns_to_keep:
    col_index = columns_to_keep.index(column)
    for i in range(len(df[column])):
        if(df[column][i] == '.'):
            df[column][i] = round((average[col_index] + (np.random.uniform(-1, 1) * std[col_index])), 4)


df = df.groupby('PATNO').first().reset_index()
df = df[columns_to_keep]
df.drop(columns=df.columns[0], axis=1, inplace=True)

#print(average[13])
#print(std[13])


#df = df.dropna(subset=columns_to_keep)
print(df.head)
df.to_csv("data/PPMI/PPMI_cleaned.csv")