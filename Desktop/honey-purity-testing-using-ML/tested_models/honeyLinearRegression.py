import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML-honey-mam\dataset\honey_purity_dataset.csv')

X = data[['CS', 'EC']]
y = data['Purity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean squared Error: {mse:.2f}')
print(f'R-square: {r2:.2f}')

plt.scatter(X_test['CS'], y_test, label='Actual')
plt.scatter(X_test['CS'], y_pred, label='Predicted')
plt.xlabel('CS')
plt.ylabel('Purity')
plt.legend()
plt.show()
