
# Logistic Regression Project — Overfitting and Regularization

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

## 📈 Run the Code
```bash
python logistic_regression_regularization.py
```

---

### Author
**Faezeh’s ML Learning Project** ✨  
A simple and practical example of Logistic Regression with Regularization.
