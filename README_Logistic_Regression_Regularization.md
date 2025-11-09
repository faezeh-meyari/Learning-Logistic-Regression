
# 🧠 01_Simple Logistic Regression Example

## 📘 Overview
This project demonstrates **the basic concept of Logistic Regression** using a simple dataset that predicts whether a student will be admitted based on their exam score.  
It also visualizes the **sigmoid curve**, which is the core function of logistic regression.

---

## 🚀 Objective
To show how logistic regression transforms input scores into probabilities and makes binary predictions (0 or 1).

---

## 🧩 Steps in the Project

### 1. Import Libraries
We use:
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Create a Simple Dataset
```python
data = {'Score': [35, 40, 50, 55, 60, 65, 70, 75, 80, 85],
        'Admit': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]}
df = pd.DataFrame(data)
```
Each score corresponds to whether the student was admitted (`1`) or not (`0`).

### 3. Train the Logistic Regression Model
```python
model = LogisticRegression()
model.fit(X, Y)
```
We train a simple model on one feature (`Score`).

### 4. Predict Probabilities and Classes
```python
probabilities = model.predict_proba(X)[:, 1]
predictions = model.predict(X)
```
- `predict_proba`: gives the probability of being in class `1`  
- `predict`: gives the final class label (0 or 1)

### 5. Visualize the Sigmoid Curve
We use matplotlib to show how probabilities change smoothly between 0 and 1.

```python
plt.plot(scores_range, probabilities_range, label='Sigmoid Curve')
plt.scatter(X, Y, color='red', label='Actual Data')
plt.axhline(0.5, color='green', linestyle='--', label='Threshold 0.5')
plt.show()
```

---

## 📈 Results
- The sigmoid curve shows how logistic regression outputs probabilities between 0 and 1.  
- The threshold (0.5) determines the final classification boundary.

---

## 🧠 Key Concepts Learned
- Logistic regression predicts **probabilities**, not just classes.
- The **sigmoid function** is central to converting linear scores into probabilities.
- Visualization helps understand model behavior intuitively.

---

## 🛠️ Requirements
- Python 3.x  
- pandas  
- scikit-learn  
- matplotlib  
- numpy

---
# 02_Logistic Regression — Train-Test Split and Model Evaluation

## 📘 Overview
This project extends the first example by showing how to **split data into training and testing sets** and evaluate the model using **accuracy**, **confusion matrix**, and **log loss**.

---

## 🚀 Objective
To demonstrate good machine learning practice — **train on one subset of data and test on another** to check generalization.

---

## 🧩 Steps in the Project

### 1. Create a Synthetic Dataset
We simulate student scores and admission results.

```python
np.random.seed(42)
scores = np.random.randint(30, 101, 100)
admit = (scores >= 60).astype(int)
df = pd.DataFrame({'Score': scores, 'Admit': admit})
```

### 2. Split into Train and Test Sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80% data for training  
- 20% data for testing

### 3. Train the Logistic Regression Model
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 4. Predict and Evaluate
```python
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
loss = log_loss(y_test, model.predict_proba(X_test))
```

---

## 📈 Results Example
```
Accuracy: 0.9
Confusion Matrix:
[[12  1]
 [ 1  6]]
Log Loss on Test set: 0.31
```

### ✅ Interpretation
- **Accuracy**: Percentage of correct predictions  
- **Confusion Matrix**: How many 0s and 1s were correctly or incorrectly predicted  
- **Log Loss**: Measures how well predicted probabilities match the true labels  
  → Lower value = better performance

---

## 🧠 Key Concepts Learned
- Importance of **train-test split** to avoid overfitting  
- How to interpret **accuracy**, **confusion matrix**, and **log loss**  
- Logistic Regression can output both **probabilities** and **classes**

---

## 🛠️ Requirements
- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
---
# 03_Logistic Regression Project — Overfitting and Regularization

## 📘 Overview
This project demonstrates how **overfitting** can occur in Logistic Regression and how **regularization** helps improve generalization.  
We create a synthetic dataset, fit an overfitted model, then apply **L2 regularization** to fix it and compare results.

---

## ⚙️ Steps

### 1. Create a Synthetic Dataset
We generate a **non-linear dataset** using `make_moons()` and add noise to make the classification challenging.

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
```

### 2. Create Overfitting
We add **polynomial features** to make the model too complex and fit a model **without regularization**.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(X)
```

Then fit Logistic Regression without regularization:
```python
from sklearn.linear_model import LogisticRegression
model_overfit = LogisticRegression(penalty=None, solver='lbfgs')
model_overfit.fit(X_poly, y)
```

### 3. Evaluate Overfitting
We check accuracy and log loss for both train and test data.
You should see **very high train accuracy** and **low test accuracy**, showing overfitting.

### 4. Apply Regularization
Now apply **L2 regularization** (Ridge-style) to limit model complexity.

```python
model_reg = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000)
model_reg.fit(X_poly, y)
```

### 5. Evaluate Again
After regularization, you should see **balanced accuracy** and **lower test log loss**.

### 6. Plot Decision Boundaries
Visualize the model performance before and after regularization using matplotlib.

---

## 📊 Example Results

| Model | Train Accuracy | Test Accuracy | Train Log Loss | Test Log Loss |
|--------|----------------|----------------|----------------|----------------|
| Overfit | 1.00 | 0.77 | 0.0003 | 8.02 |
| Regularized | 0.87 | 0.83 | 0.21 | 0.89 |

✅ Regularization improved generalization by reducing the test loss.

---

## 🧠 Key Takeaways
- **Overfitting** happens when a model learns noise instead of signal.
- **Regularization** adds a penalty on large weights to simplify the model.
- **Log Loss** is a better metric for probabilistic classifiers than accuracy.
- You can tune regularization strength with the parameter `C` (smaller `C` → stronger regularization).

---

## 🛠️ Technologies Used
- Python 3
- Scikit-learn
- NumPy
- Matplotlib

---

### Author
**Faezeh Meyari** ✨  
A simple and practical example of Logistic Regression with Regularization.
