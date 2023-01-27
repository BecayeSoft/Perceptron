import Perceptron as p
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection._split import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)
print(df.head())


# Splitting the data
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

train_data, test_data, train_labels, test_labels = train_test_split(
                            X, y, test_size=0.25)

train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

print('Train data:', train_data[0:2])
print('Train labels:', train_labels[0:5])

print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:5])

# fitting the perceptron
perceptron = p.Perceptron(eta=0.1, n_iter=10)
perceptron.fit(train_data, train_labels)

#  Predicting the results

test_preds = perceptron.predict(test_data)

print(test_preds)

# Mesuring Performances
accuracy = accuracy_score(test_preds, test_labels)
print('Accuracy:', round(accuracy, 2) * 100, "%")
