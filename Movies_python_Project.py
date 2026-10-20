import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12, 8)

# Read in the data
df = pd.read_csv('/Users/tanishkhurana/Downloads/movies.csv')

#Lets look at the data
#print(df.head())

# Finding missing data

for col in df.columns:
    np.mean(df[col].isnull())

# Data tyoes for our columns
#df.dtypes
# Remove rows with missing budget or gross
df_clean = df.dropna(subset=['budget', 'gross']).copy()

# Convert data types
df_clean['budget'] = df_clean['budget'].astype('int64')
df_clean['gross'] = df_clean['gross'].astype('int64')

# Extract release year
df_clean['correct_year'] = df_clean['released'].str.extract(r'(\d{4})')

# Sort movies by gross revenue
df_clean = df_clean.sort_values(by='gross', ascending=False)

# Drop any duplicates
df['company'].drop_duplicates().sort_values(ascending = False)

#Scattterplot with budget vs gross

plt.scatter(x=df_clean['budget'], y=df_clean['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for films')
plt.show()

#Plot budget vs gross using seaborn
sns.regplot(x='budget', y='gross', data=df_clean, scatter_kws={"color":"red"}, line_kws={"color":"blue"})
plt.title('Budget vs Gross Earnings')
plt.xlabel('Movie Budget')
plt.ylabel('Gross Earnings')
plt.show()

# Looking at the correlation
print(df_clean.corr(method = 'pearson', numeric_only=True))

# High correlation between budget and gross

correlation_matrix = df_clean.corr(method = 'pearson', numeric_only=True)

sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

# Analysing Company
df_numerised = df_clean.copy()

for col_name in df_numerised.select_dtypes(
    include=['object', 'string', 'category']
).columns:
    df_numerised[col_name] = (
        df_numerised[col_name]
        .astype('category')
        .cat.codes
    )

print(df_numerised.head())

correlation_matrix = df_numerised.corr(method = 'pearson', numeric_only=True)

sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

correlation_mat = df_numerised.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs

sorted_pairs = corr_pairs.sort_values()
print(sorted_pairs)