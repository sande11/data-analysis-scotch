import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel("D:/data analysis scotch/stata.xls", sheet_name="Sheet1")  
print(data.head())  

# Ensure 'status' column is binary (e.g., 'Withdrawn' = 1, 'Not withdrawn' = 0)
data['status'] = data['status'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

# Encode categorical variables (binary or label encoding)
le = LabelEncoder()
data['side_effects'] = le.fit_transform(data['side_effects'])  # Encode 'Yes'/'No' as 1/0
data['parent_education'] = le.fit_transform(data['parent_education'])  # Encode education levels
data['parents_occupation'] = le.fit_transform(data['parents_occupation'])  # Encode occupation categories
data['knowledge_wifa'] = le.fit_transform(data['knowledge_wifa'])  # Encode knowledge_wifa (Yes/No)

# Add constant for the intercept term
data['intercept'] = 1

# Specify independent variables (predictors)
independent_vars = ['intercept', 'side_effects', 'parent_education', 'parents_occupation', 'knowledge_wifa']

# Fit the logistic regression model
logit_model = sm.Logit(data['status'], data[independent_vars])  # Dependent variable is 'status'
result = logit_model.fit()

# Display the regression results
print(result.summary())

# Calculate Odds Ratios (ORs) by exponentiating the coefficients
odds_ratios = np.exp(result.params)

# Calculate the 95% confidence intervals for the odds ratios
conf = result.conf_int()
conf['OR'] = odds_ratios
conf.columns = ['2.5%', '97.5%', 'OR']

# Display the odds ratios and their confidence intervals
print(conf)

# Export the results to an Excel file
with pd.ExcelWriter('logistic_regression_results.xlsx') as writer:
    result_summary = pd.DataFrame(result.summary2().tables[1])  # Regression summary table
    result_summary.to_excel(writer, sheet_name='Regression Summary')
    conf.to_excel(writer, sheet_name='Odds Ratios')  # Odds ratios with confidence intervals

# Optionally, you can also export the full dataset with predicted probabilities
data['predicted_prob'] = result.predict(data[independent_vars])  # Predicted probabilities
data.to_excel('data_with_predictions.xlsx', index=False)  # Export data with predictions

print("Results exported successfully!")
