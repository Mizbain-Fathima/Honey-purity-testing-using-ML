import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML-honey-mam\dataset\honey_purity_dataset.csv')
print(data.head())
print(data.info())
print(data.describe())

print(data.isnull().sum())
data = data.fillna(data.mean(), inplace=True)
data = pd.get_dummies(data, columns=['Pollen_analysis'], prefix='', prefix_sep='')

data.hist(figsize = (12,8), bins=20)
plt.show()

corr_mat = data.corr()
plt.figure(figsize=(10,8))
plt.imshow(corr_mat, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(np.arange(len(corr_mat.columns)), corr_mat.columns, rotation=45)
plt.yticks(np.arange(len(corr_mat.columns)), corr_mat.columns)
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
data['Pollen_analysis'].value_counts().plot(kind='bar')
plt.title('Count of Pollen Analysis')
plt.xlabel('Pollen Analysis')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
data.boxplot(column='density', by='Pollen_analysis')
plt.title('Boxplot of Density by Pollen Analysis')
plt.suptitle('')  # Suppress the default title
plt.xlabel('Pollen Analysis')
plt.ylabel('Density')
plt.xticks(rotation=45)
plt.show()
