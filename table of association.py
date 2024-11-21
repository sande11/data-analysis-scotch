import pandas as pd
from scipy.stats import chi2_contingency

# Step 1: Load your data from an Excel file
df = pd.read_excel(r'D:\data analysis scotch\stata.xls')

# Step 2: Display the first few rows to understand the data
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Cross-tabulation for categorical variables
categorical_vars = ['parents_occupation', 'parent_education', 'side_effects', 'knowledge_wifa', 'status']
results = []  # List to store results

# Loop through categorical variables to create pairwise association tables
for var1 in categorical_vars:
    for var2 in categorical_vars:
        if var1 != var2:
            # Create a contingency table
            contingency_table = pd.crosstab(df[var1], df[var2])
            # Perform Chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # Add results to the list
            results.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Chi-Square': chi2,
                'p-value': p
            })

# Convert results list to a DataFrame
assoc_table = pd.DataFrame(results)

# Display the association table
print("\nAssociation Table between Categorical Variables:")
print(assoc_table)

# Step 4: Summary statistics for numeric variables
summary_stats = df.groupby(categorical_vars).agg({'age': ['mean', 'std', 'count']}).reset_index()

print("\nSummary Statistics for Age by Categorical Groups:")
print(summary_stats)
