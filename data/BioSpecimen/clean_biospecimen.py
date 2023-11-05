import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data/BioSpecimen/Current_Biospecimen_Analysis.csv')

duplicate_counts = df.groupby(["PATNO","SEX","COHORT","CLINICAL_EVENT","TYPE","TESTNAME", "TESTVALUE","UNITS","RUNDATE","PROJECTID","PI_NAME","PI_INSTITUTION","update_stamp"]).size().reset_index(name='DUPLICATE_COUNT')

duplicates_greater_than_1 = duplicate_counts[duplicate_counts['DUPLICATE_COUNT'] > 1]

# Print the duplicates
print(duplicates_greater_than_1)
# Select the columns you want to keep
columns_to_keep = ['PATNO', 'SEX', 'COHORT', 'CLINICAL_EVENT', 'TYPE', 'TESTNAME', 'TESTVALUE']

df = df[columns_to_keep]

# Pivot the data to have 'TESTNAME' columns
df_pivoted = df.pivot_table(index=['PATNO', 'SEX', 'COHORT', 'CLINICAL_EVENT'], columns='TESTNAME', values='TESTVALUE', aggfunc="first").reset_index()

# Reset column names
df_pivoted.columns.name = None

# Fill NaN values with 'NA'
#df_pivoted = df_pivoted.fillna('NA')

# Save the result to a new CSV file
df_pivoted.to_csv('data/BioSpecimen/prepped_specimen.csv', index=False)
