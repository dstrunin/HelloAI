import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X = X / 255 # Scale pixel values to 0-1 range

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

print(f"Accuracy on Training Set: {clf.score(X_train, y_train):.3f}")
print(f"Accuracy on Test Set: {clf.score(X_test, y_test):.3f}")

import numpy as np
idx = np.random.randint(0, X_test.shape[0])
img = X_test[idx].reshape(28,28)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
print(f"Prediction: {clf.predict([X_test[idx]])}")