import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, recall_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML-honey-mam\dataset\honey_purity_dataset.csv')
threshold = 0.7
data['Purity_binary'] = data['Purity'].apply(lambda x: 1 if x>=threshold else 0)

scaler = StandardScaler()
data = pd.get_dummies(data, columns=['Pollen_analysis'])
data[['CS', 'Density', 'WC', 'pH', 'F', 'G', 'Viscosity', 'EC', 'Price']] = scaler.fit_transform(data[['CS', 'Density', 'WC', 'pH', 'F', 'G', 'Viscosity', 'EC', 'Price']])

X = data.drop(['Purity', 'Purity_binary'], axis=1)
y = data['Purity_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Classification report")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
print("Accuracy:", accuracy_score(y_test, y_pred))
conf = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n ", conf)
print("Recall score: ", recall_score(y_test, y_pred))
print("Precision score: ", precision_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))