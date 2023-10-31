import pandas as pd
import numpy as np
df = pd.read_csv("data/PPMI/PPMI_Curated_Data_Cut_Public_20230612_rev.csv")

columns_to_keep = ['PATNO', 'COHORT','age_at_visit', 'SEX', 'race', 'BMI', 'moca', 'fampd', 'bjlot' ]
df[columns_to_keep] = df[columns_to_keep].replace('.', np.nan)
df = df.groupby('PATNO').first().reset_index()
df = df[columns_to_keep]

df = df.dropna(subset=columns_to_keep)
print(df.head)
df.to_csv("data/PPMI/cleaned.csv")