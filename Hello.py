import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = load_digits()
X, y = mnist.data, mnist.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
clf = LogisticRegression(max_iter=10000, solver='saga')
clf.fit(X_train, y_train)

# Evaluate model performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy:.3f}")

# Visualize a random test sample
import numpy as np
idx = np.random.randint(0, X_test.shape[0])
img = X_test[idx].reshape(8, 8)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
print(f"Prediction: {clf.predict([X_test[idx]])[0]}")