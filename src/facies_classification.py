import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.tree import DecisionTreeClassifier as Model
from sklearn.ensemble import RandomForestClassifier as Model
# from sklearn.neighbors import KNeighborsClassifier as Model
# from sklearn.linear_model import LogisticRegression as Model
# from sklearn.neural_network import MLPClassifier as Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer

def transform_X(X):
	X = np.copy(X)
	X[:,-1] = X[:,-1].astype(np.int32)
	X[:,-2] = X[:,-2].astype(np.int32)
	litho = X[:,-1]
	if 4 in litho:
		X[:,-1][litho == 4] = 3

	litho_enc = OneHotEncoder(sparse=False, categories=[[1,2,3]])
	shape_enc = OneHotEncoder(sparse=False, categories=[[1,2,3,4]])
	ct = ColumnTransformer([('litho', litho_enc, [-1]),
							('gr_shape', shape_enc, [-2])],
						   remainder='passthrough',
						   sparse_threshold=0)
	X = ct.fit_transform(X)

	return X

def predict(model, X):
	X = np.copy(X)
	litho = X[:,-1]
	X = transform_X(X)

	pred = model.predict(X)

	if 4 in litho:
		pred[litho == 4] = 5

	return pred

features = ['GR', 'DTC', 'DENS', 'NEUT', 'Thickness', 'GR_SHAPE', 'Lithofacies']
target = ['Facies_code']
df = pd.read_csv('CHEAL-1-Thickness.csv', header=0)[features + ['Facies_code']]
df = df[~((df['Lithofacies'] == 4) & (df['Facies_code'] != 5))]
df = df[~((df['Lithofacies'] != 4) & (df['Facies_code'] == 5))]
data = df[~df.isin(['-999', -999]).any(axis=1)]

X = data[features].values
y = data[target].values.astype(np.int32).flatten()

X_train, X_test, y_train, y_test = (
	train_test_split(X, y, stratify=X[:,-1], test_size=0.2, shuffle=True))

model = Model(
			  n_estimators=30,
			  criterion='entropy',
			  min_samples_split=3,
			  min_impurity_decrease=0.001,
			  class_weight='balanced')
# model = Model(n_neighbors=3, metric='minkowski', p=1)

mask = X_train[:,-1] != 4
model.fit(transform_X(X_train[mask,:]), y_train[mask])

print(model.score(transform_X(X_train[mask]), y_train[mask]))

pred = predict(model, X_train)
print(np.count_nonzero(pred == y_train) / len(pred))
print(confusion_matrix(y_train, pred))

pred = predict(model, X_test)
print(np.count_nonzero(pred == y_test) / len(pred))
print(confusion_matrix(y_test, pred))

df = pd.read_csv('CHEAL-A8.csv', header=0)
mark =df['Unit_mark_final'].values
data = df[features]
# data = df[~df.isin(['-999', -999]).any(axis=1)]
X = data.values
pred = predict(model, X)
start = 0

from scipy import stats
for i in range(len(pred)):
	if mark[i] == 1:
		tmp = pred[start:i+1]
		pred[start:i+1] = stats.mode(tmp)[0][0]
		start = i+1

df['Facies_code'] = pred
df.to_csv('CHEAL-A8.csv')
