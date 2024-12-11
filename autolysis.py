# /// script
# dependencies = [
#   "pandas",
#   "requests",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
# ]
# ///

import os, sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

file_name = sys.argv[1]
df = pd.read_csv(file_name, encoding='ISO-8859-1')

# Util funnctions
def get_number_of_outliers_per_column(df):
    """
    Calculates the number of outliers in each numerical column of a Pandas DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary where keys are column names and values are the number of outliers in each column.
    """

    num_cols = df.select_dtypes(include=np.number).columns
    outlier_counts = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    return outlier_counts

def query_model(prompt):
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    token = os.environ["AIPROXY_TOKEN"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(token), 
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code > 300:
        return None
    return response.json()["choices"][0]["message"]["content"]

# Begin Analysis
analysis_map = {}

# Columns
numeric_columns = df.select_dtypes(include='number').columns.to_list()
object_columns = df.select_dtypes(exclude='object').columns.to_list()
analysis_map['All Columns'] = df.columns.to_list()
analysis_map['Numric Columns'] = numeric_columns
analysis_map['Object Columns'] = object_columns

# General Overview
analysis_map['Summary Statistics'] = df.describe().to_dict(orient="records")
analysis_map['Missing values count per column'] = df.isnull().sum().to_dict()

# Outlier Detection
analysis_map['Outlier count per column'] = get_number_of_outliers_per_column(df)

# Correlation Analysis
correlation_matrix = df[numeric_columns].corr()
high_correlations = correlation_matrix.unstack().sort_values(ascending=False)
high_correlations = high_correlations[high_correlations < 1]
strong_positive_corr = high_correlations[high_correlations > 0.7]
strong_negative_corr = high_correlations[high_correlations < -0.7]
high_correlation_map = {}
for pair, corr in strong_positive_corr.items():
    high_correlation_map[pair] = corr
for pair, corr in strong_negative_corr.items():
    high_correlation_map[pair] = corr
analysis_map['Pair of columns with high correlation and their correlation value'] = high_correlation_map

# Plot and save heatmap
heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, cbar=True)
plt.savefig('heatmap.png', dpi=100)
plt.close()

# Feature importance Analysis
numeric_col_data = df[numeric_columns]
numeric_col_data.fillna(numeric_col_data.mean(), inplace=True)
prompt_for_feature_selection = f"""
A dataframe has following numeric columns {numeric_columns}, I want to know the name of a possible target column if it exists.
Please only output the name of the most probable target column or NF if nothing found
"""
target_column = query_model(prompt_for_feature_selection)
if target_column != 'NF':
    X = df.drop(columns=[target_column]).select_dtypes(include=['number'])
    X = X.fillna(X.mean())
    y = df[target_column].fillna(df[target_column].mean())
    model = RandomForestRegressor(random_state=0)
    model.fit(X, y)
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance'])
    analysis_map["Feature Importance"] = feature_importances.sort_values(by="Importance", ascending=False).to_dict()

# Time Series Analysis
prompt_for_timeseries = f"""
A dataframe has following columns {df.columns.to_list()}, I want to do a time series analysis per month on this. Which is the fittest date/time column
in this dataset. Please only output the name of the most probable target column or NF if nothing found
 """
ts_column = query_model(prompt_for_timeseries)
if ts_column != 'NF':
    df[ts_column] = pd.to_datetime(df[ts_column])
    month_counts = df[ts_column].dt.to_period('M').value_counts().sort_index()
    month_counts_df = month_counts.reset_index()
    month_counts_df.columns = ['Month', 'Counts']
    analysis_map['Data distribution per month'] = month_counts_df.to_dict()


# Cluster Analysis
k_values = range(2, 11)
selected_sil_score = -1
selected_k = 0
# Calculate silhouette scores for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    score = silhouette_score(X, cluster_labels)
    if score > selected_sil_score:
        selected_k = k
        selected_sil_score = score
kmeans = KMeans(n_clusters=selected_k, random_state=0).fit(numeric_col_data)
df['Cluster'] = kmeans.labels_
cluter_value_counts = df['Cluster'].value_counts()
analysis_map["Cluster distrebution"] = cluter_value_counts.to_dict()

plt.figure(figsize=(6, 6))
cluter_value_counts.plot.pie(
    autopct='%1.1f%%',  
    startangle=90
)
plt.title(f"Distribution of Clusters")
plt.ylabel('')
plt.tight_layout()
plt.savefig('cluster_pie.png', dpi=100, bbox_inches='tight')
plt.close()

    
prompt_for_story = f"""
I have a dataset with the following characteristics:
- Number of rows: {df.shape[0]}
- Number of columns: {df.shape[1]}
- Column names: {analysis_map['All Columns']}
- Data types of columns: {df.dtypes.to_dict()}
- Numeric column names: {analysis_map['Numric Columns']}
- Object column names: {analysis_map['Object Columns']}
- Summary statistics: {analysis_map['Summary Statistics']}
- Missing values per column: {analysis_map['Missing values count per column']}
- Outliers per column: {analysis_map['Outlier count per column']}
- Memory usage: {df.memory_usage(deep=True).sum()}
- Highly correlated columns - a dictionary witk key as tuple of column names and value as correlation value: {analysis_map['Pair of columns with high correlation and their correlation value'] }
- Best clustering distribution: {analysis_map["Cluster distrebution"]}
"""

if "Feature Importance" in analysis_map:
    f_imp_str = f"""  - Feature importance : {analysis_map["Feature Importance"]} """
    prompt_for_story = prompt_for_story +"\n"+"\n" + f_imp_str

if "Data distribution per month" in analysis_map:
    month_dist_str = f"""  - Data distribution per month (Count of rows per month): {analysis_map['Data distribution per month']}"""
    prompt_for_story = prompt_for_story+"\n"+"\n"+ month_dist_str
   
end_str = f"""
Here is a sample of the data:
{df.head()}

I want to output text in markdown format with following bullet points from the charecteristics of data I have provided above. 
Also please point me to any additional charecteristics I can provide you to improve your story.

1. Brief description of the dataset (Try include the type of data - eg: Crop cultivation data or Sales data ...)
2. Comments on the analysis above (Also any suggestions for the Analysis - do not include the text in current bracket in title of markdown text)
3. Insights from the data set
4. What can we do with the findings
"""

prompt_for_story = prompt_for_story+"\n"+end_str

result = query_model(prompt_for_story)
if result.splitlines()[0] == '```markdown':
  result = result.split('\n',1)[-1]

output_file = "README.md"
with open(output_file, "w") as file:
    file.write(result)




















