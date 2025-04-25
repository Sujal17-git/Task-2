# 1. Generate summary statistics.

import pandas as pd 

df = pd.read_csv('Titanic-Dataset.csv')
df.describe()

# 2.Create histograms and boxplots for numeric features.

import matplotlib.pyplot as plt
import seaborn as sns

df['Age'].hist(figsize=(10, 8), bins=20, edgecolor='black')
plt.suptitle("Histogram of Numeric Features", fontsize=14)
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(data=df[['Age','Fare']])  
plt.title("Boxplot of Age & Fare")
plt.show()

# 3.Use pairplot/correlation matrix for feature relationships.


sns.pairplot(df[['Age', 'Fare']])
plt.show()

numeric_df = df.select_dtypes(include=['number'])  # Keeps only number-type columns

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")  
plt.title("Correlation Matrix (Numeric Features Only)")
plt.show()

# 4.Identify patterns, trends, or anomalies in the data.

sns.barplot(x='Pclass', y='Survived', data=df, errorbar=None)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None)
plt.title("Survival Rate by Gender")
plt.show()
