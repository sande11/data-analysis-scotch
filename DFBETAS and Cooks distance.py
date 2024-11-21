import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load your dataset from an Excel file (stata.xls)
data = pd.read_excel("D:\data analysis scotch\stata.xls")

# Handling missing values as per previous context
data = data.assign(
    age=data['age'].fillna(data['age'].mean()),
    side_effects=data['side_effects'].fillna(data['side_effects'].mean()),
    parent_education=data['parent_education'].fillna(data['parent_education'].mean()),
    parents_occupation=data['parents_occupation'].fillna(data['parents_occupation'].mean()),
    knowledge_wifa=data['knowledge_wifa'].fillna(data['knowledge_wifa'].mean())
)

# Remove any remaining rows with NaNs or infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Define your dependent (y) variable and independent (X) variables
# Here, we are using 'side_effects' as the dependent variable
y = data['side_effects']  # Assuming 'side_effects' is a binary outcome
X = data[['age', 'parents_occupation', 'parent_education', 'knowledge_wifa']]
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Calculate DFBetas
influence = result.get_influence()
dfbetas = influence.dfbetas

# Calculate Cook's Distance
cooks_d = influence.cooks_distance[0]

# Create a DataFrame to display DFBetas and Cook's Distance
dfbetas_df = pd.DataFrame(dfbetas, columns=[f'DFBetas_{col}' for col in X.columns])
cooks_distance_df = pd.DataFrame(cooks_d, columns=['Cook_Distance'])

# Combine DFBetas and Cook's Distance into one DataFrame
influence_df = pd.concat([dfbetas_df, cooks_distance_df], axis=1)

# Display the influence table
print(influence_df)
