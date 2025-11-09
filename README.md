# Machine Learning

# Table of contents

- [Introduction](#introduction)
- [Supervised Learning](#supervised-learning)
- [Training Models](#training-models)

# Introduction

# Supervised Learning

## Training Models

### Linear Regression

Equation for Linear Regression model prediction

$$
\hat y=\theta_0+\theta_1 x_1+\theta_2 x_2+\dots +\theta_n x_n=h_\theta(\vec x)=\vec \theta\cdot\vec x
$$

## 1. Mathematical Foundation

### Core Equation

```
y = mx + b  (simple)
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ  (multiple)
```

**Matrix Form:**

```
ŷ = Xβ
where:
  ŷ = predictions (m × 1)
  X = feature matrix (m × n+1, includes bias column)
  β = weights/coefficients (n+1 × 1)
```

### Cost Function (Mean Squared Error)

```
J(β) = (1/2m) Σ(ŷᵢ - yᵢ)²
     = (1/2m) ||Xβ - y||²
```

### Gradient (Derivative of Cost)

```
∇J(β) = (1/m) Xᵀ(Xβ - y)
```

-----

## 2. Implementation Approaches

### Method A: Normal Equation (Closed-Form Solution)

**Formula:**

```
β = (XᵀX)⁻¹Xᵀy
```

**Pros:** Direct solution, no hyperparameters  
**Cons:** Slow for large datasets (O(n³)), requires matrix inversion

### Method B: Gradient Descent (Iterative)

**Update Rule:**

```
β := β - α∇J(β)
β := β - α(1/m)Xᵀ(Xβ - y)
```

where α is the learning rate

**Pros:** Works with large datasets, scalable  
**Cons:** Requires tuning learning rate, may need many iterations

-----

## 3. Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression implemented from scratch.
    Supports both Normal Equation and Gradient Descent.
    """
    
    def __init__(self, method='gradient_descent', learning_rate=0.01, 
                 n_iterations=1000, regularization=None, lambda_=0.01):
        """
        Parameters:
        -----------
        method : str
            'normal_equation' or 'gradient_descent'
        learning_rate : float
            Step size for gradient descent (alpha)
        n_iterations : int
            Number of iterations for gradient descent
        regularization : str or None
            'l1' (Lasso), 'l2' (Ridge), or None
        lambda_ : float
            Regularization strength
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _add_bias(self, X):
        """Add bias column (column of ones) to feature matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _compute_cost(self, X, y):
        """Calculate MSE cost function."""
        m = len(y)
        predictions = X @ self.theta
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        
        # Add regularization term
        if self.regularization == 'l2':
            # Don't regularize bias term
            cost += (self.lambda_/(2*m)) * np.sum(self.theta[1:]**2)
        elif self.regularization == 'l1':
            cost += (self.lambda_/m) * np.sum(np.abs(self.theta[1:]))
        
        return cost
    
    def fit(self, X, y):
        """
        Train the linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Target values
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Add bias term
        X_b = self._add_bias(X)
        m, n = X_b.shape
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X_b, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X_b, y)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
        
        # Separate bias and weights
        self.bias = self.theta[0, 0]
        self.weights = self.theta[1:].flatten()
        
        return self
    
    def _fit_normal_equation(self, X, y):
        """Fit using closed-form solution."""
        if self.regularization == 'l2':
            # Ridge regression
            identity = np.eye(X.shape[1])
            identity[0, 0] = 0  # Don't regularize bias
            self.theta = np.linalg.inv(X.T @ X + self.lambda_ * identity) @ X.T @ y
        else:
            # Standard OLS
            self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent optimization."""
        m, n = X.shape
        self.theta = np.zeros((n, 1))
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Compute predictions
            predictions = X @ self.theta
            
            # Compute gradient
            gradient = (1/m) * X.T @ (predictions - y)
            
            # Add regularization gradient
            if self.regularization == 'l2':
                reg_gradient = (self.lambda_/m) * self.theta
                reg_gradient[0] = 0  # Don't regularize bias
                gradient += reg_gradient
            elif self.regularization == 'l1':
                reg_gradient = (self.lambda_/m) * np.sign(self.theta)
                reg_gradient[0] = 0
                gradient += reg_gradient
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Track cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Optional: Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like, shape (m, n)
            Features to predict on
            
        Returns:
        --------
        predictions : array, shape (m,)
        """
        X = np.array(X)
        X_b = self._add_bias(X)
        return (X_b @ self.theta).flatten()
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations (for gradient descent)."""
        if not self.cost_history:
            print("No cost history available. Use method='gradient_descent'")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Cost Function Over Iterations')
        plt.grid(True)
        plt.show()
```

-----

## 4. Usage Examples

### Example 1: Simple Linear Regression

```python
# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Method 1: Normal Equation
model_ne = LinearRegression(method='normal_equation')
model_ne.fit(X, y)
print(f"Weights: {model_ne.weights}, Bias: {model_ne.bias}")

# Method 2: Gradient Descent
model_gd = LinearRegression(method='gradient_descent', 
                            learning_rate=0.1, 
                            n_iterations=1000)
model_gd.fit(X, y)
print(f"Weights: {model_gd.weights}, Bias: {model_gd.bias}")

# Make predictions
X_new = np.array([[0], [2]])
predictions = model_gd.predict(X_new)
print(f"Predictions: {predictions}")

# Calculate R² score
r2 = model_gd.score(X, y)
print(f"R² Score: {r2:.4f}")
```

### Example 2: Multiple Linear Regression

```python
# Multiple features
X = np.random.rand(100, 3)
y = 4 + 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

model = LinearRegression(method='gradient_descent', 
                        learning_rate=0.1, 
                        n_iterations=1000)
model.fit(X, y)

print(f"Coefficients: {model.weights}")
print(f"Intercept: {model.bias}")
print(f"R² Score: {model.score(X, y):.4f}")
```

### Example 3: Ridge Regression (L2 Regularization)

```python
model_ridge = LinearRegression(method='gradient_descent',
                               learning_rate=0.1,
                               n_iterations=1000,
                               regularization='l2',
                               lambda_=0.1)
model_ridge.fit(X, y)
```

-----

## 5. Key Concepts & Best Practices

### Feature Scaling

Always scale features when using gradient descent:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Learning Rate Selection

- Too high: Cost oscillates or diverges
- Too low: Slow convergence
- Start with 0.01, adjust based on cost plot

### Convergence Checking

```python
# Stop if change in cost is small
if len(model.cost_history) > 1:
    if abs(model.cost_history[-1] - model.cost_history[-2]) < 1e-6:
        break
```

### Train/Test Split

```python
# Split data 80/20
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
```

-----

## 6. Common Pitfalls & Solutions

|Problem              |Cause                 |Solution                                        |
|---------------------|----------------------|------------------------------------------------|
|Cost increasing      |Learning rate too high|Reduce α (e.g., 0.1 → 0.01)                     |
|Slow convergence     |Learning rate too low |Increase α or use more iterations               |
|Poor predictions     |Features not scaled   |Apply StandardScaler                            |
|Singular matrix error|Multicollinearity     |Remove correlated features or use regularization|
|Overfitting          |Too many features     |Use L1/L2 regularization                        |

-----

## 7. Evaluation Metrics

### R² Score (Coefficient of Determination)

```python
r2 = 1 - (SS_res / SS_tot)
# Range: (-∞, 1], best = 1
```

### Mean Squared Error (MSE)

```python
mse = np.mean((y_pred - y_true)**2)
```

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

### Mean Absolute Error (MAE)

```python
mae = np.mean(np.abs(y_pred - y_true))
```

-----

## 8. Advanced Topics

### Polynomial Features

```python
# Transform X to include polynomial terms
X_poly = np.c_[X, X**2, X**3]
model.fit(X_poly, y)
```

### Batch vs Stochastic Gradient Descent

```python
# Mini-batch gradient descent
batch_size = 32
for epoch in range(n_epochs):
    for i in range(0, m, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        # Compute gradient on batch only
```

### Learning Rate Decay

```python
alpha_t = alpha_0 / (1 + decay_rate * t)
```

-----

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│ LINEAR REGRESSION CHEAT SHEET           │
├─────────────────────────────────────────┤
│ Hypothesis: ŷ = Xβ                      │
│ Cost: J = (1/2m)||Xβ - y||²             │
│ Gradient: ∇J = (1/m)Xᵀ(Xβ - y)          │
│ Update: β := β - α∇J                    │
│                                          │
│ NORMAL EQUATION                          │
│   β = (XᵀX)⁻¹Xᵀy                         │
│   • Fast for small n (< 10,000)         │
│   • No iterations needed                 │
│                                          │
│ GRADIENT DESCENT                         │
│   • Scale features first!                │
│   • Tune learning rate α                 │
│   • Check cost plot for convergence      │
│                                          │
│ TYPICAL WORKFLOW                         │
│   1. Load & explore data                 │
│   2. Split train/test                    │
│   3. Scale features                      │
│   4. Fit model                           │
│   5. Evaluate (R², RMSE)                 │
│   6. Predict on new data                 │
└─────────────────────────────────────────┘
```


#### Mean Squared Error (MSE)
Equation for Mean Squared Error

$$
\text{MSE}\, (\vec X,h_\theta)=\frac{1}{m}\sum_{i=1}^m(\vec \theta^T\vec x^{(i)}-y^{(i)})^2
$$

### Gradient Descent

### Regularized Linear Models

### Logistic Regression


# Logistic Regression from Scratch - Complete Reference Guide

## Overview

Logistic regression is a supervised learning algorithm for binary classification that predicts probabilities using the sigmoid function. Despite its name, it’s used for classification, not regression.

-----

## Mathematical Foundation

### 1. Sigmoid Function

Converts any real value to a probability between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

where `z = w·x + b` (linear combination of weights, features, and bias)

### 2. Hypothesis Function

```
h(x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))
```

### 3. Cost Function (Binary Cross-Entropy Loss)

```
J(w,b) = -1/m × Σ[y^(i) × log(h(x^(i))) + (1-y^(i)) × log(1-h(x^(i)))]
```

where:

- m = number of training examples
- y^(i) = actual label (0 or 1)
- h(x^(i)) = predicted probability

### 4. Gradient Descent Update Rules

```
w := w - α × ∂J/∂w
b := b - α × ∂J/∂b
```

where α is the learning rate

**Partial derivatives:**

```
∂J/∂w = 1/m × X^T × (h(X) - y)
∂J/∂b = 1/m × Σ(h(x^(i)) - y^(i))
```

-----

## Implementation Steps

### Step 1: Import Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```

### Step 2: Define the Sigmoid Function

```python
def sigmoid(z):
    """
    Compute sigmoid activation
    Args:
        z: numpy array of any shape
    Returns:
        sigmoid(z): same shape as z
    """
    return 1 / (1 + np.exp(-z))
```

### Step 3: Initialize Parameters

```python
def initialize_parameters(n_features):
    """
    Initialize weights and bias to zeros
    Args:
        n_features: number of input features
    Returns:
        w: weight vector of shape (n_features, 1)
        b: bias scalar
    """
    w = np.zeros((n_features, 1))
    b = 0
    return w, b
```

### Step 4: Compute Cost and Gradients

```python
def compute_cost_and_gradients(X, y, w, b):
    """
    Compute cost function and gradients
    Args:
        X: training data of shape (m, n_features)
        y: labels of shape (m, 1)
        w: weights of shape (n_features, 1)
        b: bias scalar
    Returns:
        cost: binary cross-entropy loss
        dw: gradient of cost w.r.t. w
        db: gradient of cost w.r.t. b
    """
    m = X.shape[0]  # number of examples
    
    # Forward propagation
    z = np.dot(X, w) + b
    A = sigmoid(z)
    
    # Compute cost (with clipping to avoid log(0))
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)
    cost = -1/m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    # Backward propagation
    dw = 1/m * np.dot(X.T, (A - y))
    db = 1/m * np.sum(A - y)
    
    return cost, dw, db
```

### Step 5: Gradient Descent Optimization

```python
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    """
    Optimize parameters using gradient descent
    Args:
        X: training data of shape (m, n_features)
        y: labels of shape (m, 1)
        w: initial weights
        b: initial bias
        learning_rate: step size for gradient descent
        num_iterations: number of optimization iterations
    Returns:
        w: optimized weights
        b: optimized bias
        costs: list of costs during training
    """
    costs = []
    
    for i in range(num_iterations):
        # Compute cost and gradients
        cost, dw, db = compute_cost_and_gradients(X, y, w, b)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return w, b, costs
```

### Step 6: Make Predictions

```python
def predict(X, w, b, threshold=0.5):
    """
    Predict binary labels
    Args:
        X: data of shape (m, n_features)
        w: trained weights
        b: trained bias
        threshold: classification threshold (default 0.5)
    Returns:
        predictions: binary predictions (0 or 1)
    """
    z = np.dot(X, w) + b
    A = sigmoid(z)
    predictions = (A >= threshold).astype(int)
    return predictions
```

### Step 7: Complete LogisticRegression Class

```python
class LogisticRegression:
    """
    Logistic Regression classifier from scratch
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.costs = []
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        Args:
            X: training data of shape (m, n_features)
            y: labels of shape (m,) or (m, 1)
        """
        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Initialize parameters
        n_features = X.shape[1]
        self.w, self.b = initialize_parameters(n_features)
        
        # Optimize using gradient descent
        self.w, self.b, self.costs = gradient_descent(
            X, y, self.w, self.b, 
            self.learning_rate, 
            self.num_iterations
        )
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary labels
        Args:
            X: data of shape (m, n_features)
            threshold: classification threshold
        Returns:
            predictions: binary predictions
        """
        return predict(X, self.w, self.b, threshold)
    
    def predict_proba(self, X):
        """
        Predict probabilities
        Args:
            X: data of shape (m, n_features)
        Returns:
            probabilities: predicted probabilities
        """
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)
    
    def plot_cost(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.costs)
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Cost')
        plt.title('Cost Function over Training')
        plt.grid(True)
        plt.show()
```

-----

## Complete Usage Example

```python
# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=2, 
    n_redundant=0, 
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for gradient descent!)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# Create and train model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot cost
model.plot_cost()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)
```

-----

## Key Implementation Tips

### 1. **Feature Scaling**

Always normalize/standardize features before training:

```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

This ensures faster convergence and numerical stability.

### 2. **Learning Rate Selection**

- Too high: divergence or oscillation
- Too low: slow convergence
- Typical values: 0.001 to 0.1
- Use learning rate decay for better results

### 3. **Avoiding Numerical Issues**

Clip sigmoid outputs to prevent log(0):

```python
A = np.clip(A, 1e-15, 1 - 1e-15)
```

### 4. **Vectorization**

Always use NumPy vectorized operations instead of loops for efficiency:

```python
# Good: vectorized
z = np.dot(X, w) + b

# Bad: loop
z = np.zeros((m, 1))
for i in range(m):
    z[i] = np.dot(X[i], w) + b
```

### 5. **Regularization (L2)**

Add penalty term to prevent overfitting:

```python
# Modified cost
cost = -1/m * np.sum(y * np.log(A) + (1-y) * np.log(1-A)) + (lambda_reg/(2*m)) * np.sum(w**2)

# Modified gradient
dw = 1/m * np.dot(X.T, (A - y)) + (lambda_reg/m) * w
```

-----

## Common Extensions

### 1. Mini-Batch Gradient Descent

```python
def mini_batch_gradient_descent(X, y, w, b, learning_rate, 
                                num_epochs, batch_size):
    m = X.shape[0]
    costs = []
    
    for epoch in range(num_epochs):
        # Shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            cost, dw, db = compute_cost_and_gradients(X_batch, y_batch, w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
        
        if epoch % 10 == 0:
            full_cost, _, _ = compute_cost_and_gradients(X, y, w, b)
            costs.append(full_cost)
    
    return w, b, costs
```

### 2. Multi-class Classification (One-vs-All)

```python
class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.classifiers = []
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        for c in self.classes:
            # Create binary labels (one-vs-all)
            y_binary = (y == c).astype(int)
            
            # Train binary classifier
            clf = LogisticRegression(self.learning_rate, self.num_iterations)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
    
    def predict(self, X):
        # Get probabilities from all classifiers
        probs = np.column_stack([clf.predict_proba(X) for clf in self.classifiers])
        
        # Return class with highest probability
        return self.classes[np.argmax(probs, axis=1)]
```

-----

## Performance Metrics

```python
from sklearn.metrics import classification_report, roc_curve, auc

# Detailed classification report
print(classification_report(y_test, y_pred_test))

# ROC Curve
y_proba = model.predict_proba(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

-----

## Comparison with scikit-learn

```python
from sklearn.linear_model import LogisticRegression as SklearnLR

# Our implementation
our_model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
our_model.fit(X_train_scaled, y_train)
our_accuracy = accuracy_score(y_test, our_model.predict(X_test_scaled))

# Scikit-learn implementation
sklearn_model = SklearnLR(max_iter=1000)
sklearn_model.fit(X_train_scaled, y_train)
sklearn_accuracy = accuracy_score(y_test, sklearn_model.predict(X_test_scaled))

print(f"Our Implementation: {our_accuracy:.4f}")
print(f"Scikit-learn: {sklearn_accuracy:.4f}")
```

-----

## Quick Reference Summary

|Component   |Formula                             |Purpose               |
|------------|------------------------------------|----------------------|
|Sigmoid     |σ(z) = 1/(1 + e^(-z))               |Convert to probability|
|Hypothesis  |h(x) = σ(wx + b)                    |Make prediction       |
|Cost        |J = -1/m Σ[y log(h) + (1-y)log(1-h)]|Measure error         |
|Gradient (w)|∂J/∂w = 1/m X^T(h - y)              |Update weights        |
|Gradient (b)|∂J/∂b = 1/m Σ(h - y)                |Update bias           |
|Update      |w := w - α∂J/∂w                     |Optimize parameters   |

**Key Hyperparameters:**

- Learning rate (α): 0.001 - 0.1
- Iterations: 1000 - 10000
- Regularization (λ): 0.01 - 10

**Prerequisites:**

- Feature scaling/normalization
- Handling missing values
- Encoding categorical variables


## Support Vector Machines

## Decision Trees

## Ensemble Learning and Random Forests

# Random Forest Regressor from Scratch - Complete Guide

## Overview

A **Random Forest Regressor** is an ensemble learning method that combines multiple decision trees to make predictions. Each tree is trained on a random subset of data and features, and the final prediction is the average of all tree predictions.

## Key Concepts

### 1. Bootstrap Aggregating (Bagging)

- Sample data with replacement to create multiple training sets
- Each tree sees a different version of the data
- Reduces variance and prevents overfitting

### 2. Feature Randomness

- At each split, consider only a random subset of features
- Typical choice: `sqrt(n_features)` or `n_features/3`
- Decorrelates trees and improves diversity

### 3. Prediction Aggregation

- For regression: average all tree predictions
- Reduces prediction variance compared to single tree

## Implementation Steps

### Step 1: Decision Tree Node Structure

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Leaf node prediction value
    
    def is_leaf(self):
        return self.value is not None
```

### Step 2: Decision Tree Regressor

```python
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
    
    def fit(self, X, y):
        """Build decision tree"""
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the tree"""
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Node(value=np.mean(y))
        
        # Split data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        # Grow children
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold,
                   left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        """Find the best split using MSE reduction"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        # Random feature selection
        feature_indices = np.random.choice(
            self.n_features, self.max_features, replace=False
        )
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                # Calculate MSE
                mse = self._calculate_mse(y[left_idx], y[right_idx])
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_mse(self, left_y, right_y):
        """Calculate weighted MSE for a split"""
        n = len(left_y) + len(right_y)
        left_mse = np.var(left_y) * len(left_y) / n
        right_mse = np.var(right_y) * len(right_y) / n
        return left_mse + right_mse
    
    def predict(self, X):
        """Predict for all samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
```

### Step 3: Random Forest Regressor

```python
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=10, 
                 min_samples_split=2, max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
    
    def fit(self, X, y):
        """Train the forest"""
        self.trees = []
        n_samples, n_features = X.shape
        
        # Determine max_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
            
            # Train tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        """Make predictions by averaging tree predictions"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

## Complete Usage Example

```python
# Generate sample data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    bootstrap=True
)

rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate
r2_score = rf.score(X_test, y_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"R² Score: {r2_score:.4f}")
print(f"MSE: {mse:.4f}")
```

## Hyperparameter Tuning Guide

|Parameter          |Description           |Typical Values     |Effect                                    |
|-------------------|----------------------|-------------------|------------------------------------------|
|`n_estimators`     |Number of trees       |100-500            |More trees = better performance but slower|
|`max_depth`        |Max tree depth        |5-20               |Deeper = more complex, risk overfitting   |
|`min_samples_split`|Min samples to split  |2-10               |Higher = more regularization              |
|`max_features`     |Features per split    |‘sqrt’, ‘log2’, int|Lower = more diversity, less correlation  |
|`bootstrap`        |Use bootstrap sampling|True/False         |True = bagging, reduces variance          |

## Performance Optimization Tips

### 1. Speed Improvements

- Use vectorized operations with NumPy
- Consider parallel tree training (use `multiprocessing`)
- Limit tree depth to reduce computation
- Cache sorted feature values

### 2. Memory Optimization

- Use sparse matrices for sparse data
- Implement tree pruning
- Store only essential node information

### 3. Accuracy Improvements

- Increase number of trees (diminishing returns after ~200)
- Tune `max_features` to balance diversity and accuracy
- Use cross-validation to find optimal `max_depth`
- Feature engineering before training

## Common Pitfalls

1. **Not shuffling data**: Always shuffle before training
1. **Wrong max_features**: Too high = correlated trees, too low = weak trees
1. **Too few trees**: Start with at least 100 trees
1. **Ignoring scaling**: While RF handles unscaled features, scaling can help with other preprocessing
1. **Memory issues with deep trees**: Limit depth for large datasets

## Comparison with Scikit-learn

```python
from sklearn.ensemble import RandomForestRegressor as SKLearnRF

# Our implementation
rf_custom = RandomForestRegressor(n_estimators=100, max_depth=10)
rf_custom.fit(X_train, y_train)
score_custom = rf_custom.score(X_test, y_test)

# Scikit-learn
rf_sklearn = SKLearnRF(n_estimators=100, max_depth=10, random_state=42)
rf_sklearn.fit(X_train, y_train)
score_sklearn = rf_sklearn.score(X_test, y_test)

print(f"Custom RF R²: {score_custom:.4f}")
print(f"Sklearn RF R²: {score_sklearn:.4f}")
```

## Extensions and Advanced Features

### Out-of-Bag (OOB) Score

```python
def oob_score(self, X, y):
    """Calculate out-of-bag score for model evaluation"""
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    
    for i, tree in enumerate(self.trees):
        # Get OOB samples (not in bootstrap sample)
        oob_indices = # Track which samples weren't used
        predictions[oob_indices] += tree.predict(X[oob_indices])
        counts[oob_indices] += 1
    
    # Average predictions
    predictions /= counts
    return 1 - np.mean((y - predictions) ** 2) / np.var(y)
```

### Feature Importance

```python
def feature_importances_(self):
    """Calculate feature importance based on split reduction"""
    importances = np.zeros(self.n_features)
    
    for tree in self.trees:
        # Traverse tree and accumulate MSE reductions
        importances += self._tree_feature_importance(tree)
    
    importances /= self.n_estimators
    return importances / np.sum(importances)
```

## Mathematical Foundation

**Variance Reduction Formula:**

```
Var(avg(X₁,...,Xₙ)) = σ²/n + (n-1)/n × ρ × σ²
```

Where:

- σ² = variance of individual tree
- ρ = correlation between trees
- n = number of trees

**Goal:** Minimize ρ through randomness while keeping σ² reasonable

**MSE Split Criterion:**

```
MSE = (n_left/n) × Var(y_left) + (n_right/n) × Var(y_right)
```

## References and Further Reading

- Breiman, L. (2001). “Random Forests”. Machine Learning 45(1): 5-32
- Hastie, T., et al. (2009). “The Elements of Statistical Learning”
- Scikit-learn documentation on Random Forests
- Understanding feature importance in Random Forests

# Unsupervised Learning

## K-Means Clustering

# K-Means Clustering: Complete Reference Guide

## Overview

K-Means is an unsupervised machine learning algorithm that partitions data into K distinct clusters based on feature similarity. Points are assigned to the cluster with the nearest centroid.

## Algorithm Steps

### 1. Initialization

- Choose K (number of clusters)
- Initialize K centroids randomly from data points or using advanced methods (K-Means++)

### 2. Assignment Step

- For each data point, calculate distance to all centroids
- Assign point to nearest centroid’s cluster

### 3. Update Step

- Recalculate centroids as the mean of all points in each cluster

### 4. Convergence Check

- Repeat steps 2-3 until centroids stop moving or max iterations reached

## Mathematical Foundation

**Distance Metric (Euclidean):**

```
d(x, y) = √(Σ(xi - yi)²)
```

**Centroid Calculation:**

```
centroid = (1/n) * Σ(points in cluster)
```

**Objective Function (minimize within-cluster variance):**

```
J = ΣΣ ||xi - μk||²
```

## Python Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        """
        Initialize K-Means clustering algorithm
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        max_iters : int
            Maximum number of iterations
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(self, X):
        """Randomly initialize centroids from data points"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Random initialization: select K random points
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def _initialize_centroids_plusplus(self, X):
        """
        K-Means++ initialization for better starting centroids
        Selects centroids that are far apart from each other
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        centroids = []
        
        # Choose first centroid randomly
        first_idx = np.random.randint(X.shape[0])
        centroids.append(X[first_idx])
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Calculate distances from points to nearest centroid
            distances = np.array([
                min([np.linalg.norm(x - c)**2 for c in centroids])
                for x in X
            ])
            
            # Choose next centroid with probability proportional to distance²
            probabilities = distances / distances.sum()
            next_idx = np.random.choice(X.shape[0], p=probabilities)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
    
    def _compute_distances(self, X, centroids):
        """
        Compute Euclidean distance between each point and all centroids
        
        Returns:
        --------
        distances : array of shape (n_samples, n_clusters)
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(centroids):
            # Calculate distance from each point to this centroid
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return distances
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Calculate new centroids as mean of assigned points"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Calculate mean of cluster points
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster: reinitialize randomly
                centroids[k] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _has_converged(self, old_centroids, new_centroids, tolerance=1e-4):
        """Check if centroids have stopped moving"""
        return np.allclose(old_centroids, new_centroids, atol=tolerance)
    
    def fit(self, X, use_plusplus=True):
        """
        Fit K-Means to data
        
        Parameters:
        -----------
        X : array of shape (n_samples, n_features)
            Training data
        use_plusplus : bool
            Use K-Means++ initialization if True
        """
        # Initialize centroids
        if use_plusplus:
            self.centroids = self._initialize_centroids_plusplus(X)
        else:
            self.centroids = self._initialize_centroids(X)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iters):
            # Assignment step
            self.labels = self._assign_clusters(X, self.centroids)
            
            # Store old centroids
            old_centroids = self.centroids.copy()
            
            # Update step
            self.centroids = self._update_centroids(X, self.labels)
            
            # Check convergence
            if self._has_converged(old_centroids, self.centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X, use_plusplus=True):
        """Fit model and return cluster labels"""
        self.fit(X, use_plusplus)
        return self.labels
    
    def calculate_inertia(self, X):
        """
        Calculate within-cluster sum of squares (WCSS)
        Lower is better
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k])**2)
        return inertia


# Example Usage
def example_usage():
    """Demonstration of K-Means implementation"""
    
    # Generate sample data
    np.random.seed(42)
    
    # Create 3 clusters of data
    cluster1 = np.random.randn(100, 2) + [2, 2]
    cluster2 = np.random.randn(100, 2) + [8, 3]
    cluster3 = np.random.randn(100, 2) + [5, 8]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=3, max_iters=100, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate inertia
    inertia = kmeans.calculate_inertia(X)
    print(f"Inertia (WCSS): {inertia:.2f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    
    # Plot data points colored by cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot centroids
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c='red', marker='X', s=200, edgecolors='black', 
               linewidth=2, label='Centroids')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('K-Means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def elbow_method(X, max_k=10):
    """
    Use elbow method to find optimal number of clusters
    Plot inertia vs number of clusters
    """
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.calculate_inertia(X))
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method: Finding Optimal K')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return inertias


# Run example
if __name__ == "__main__":
    example_usage()
```

## Key Implementation Details

### 1. Distance Calculation

```python
# Euclidean distance between point and centroid
distance = np.linalg.norm(point - centroid)

# Vectorized for all points
distances = np.linalg.norm(X - centroid, axis=1)
```

### 2. Centroid Updates

```python
# Mean of all points in cluster k
centroid_k = X[labels == k].mean(axis=0)
```

### 3. Convergence Criteria

- Centroids don’t move (within tolerance)
- Maximum iterations reached
- No points change clusters

## Algorithm Characteristics

### Advantages

- Simple and intuitive
- Fast and efficient for large datasets
- Works well with spherical clusters
- Scales to high dimensions

### Limitations

- Requires specifying K in advance
- Sensitive to initial centroid placement
- Assumes clusters are spherical and similar size
- Sensitive to outliers
- Can converge to local minima

## Choosing Optimal K

### 1. Elbow Method

Plot inertia (WCSS) vs K and look for the “elbow” point where improvement diminishes.

### 2. Silhouette Score

```python
# Measures how similar a point is to its cluster vs other clusters
# Range: [-1, 1], higher is better
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

### 3. Domain Knowledge

Use understanding of the problem to guide K selection.

## Advanced Techniques

### K-Means++ Initialization

- Choose first centroid randomly
- Select subsequent centroids with probability proportional to distance² from nearest existing centroid
- Leads to better convergence and final clusters

### Mini-Batch K-Means

- Use random subsets of data for updates
- Faster for very large datasets
- Slight decrease in cluster quality

### Handling Empty Clusters

- Reinitialize empty cluster centroids randomly
- Split largest cluster
- Remove empty cluster and reduce K

## Common Pitfalls

1. **Not scaling features**: Features with larger ranges dominate distance calculations
1. **Ignoring outliers**: Outliers can significantly affect centroids
1. **Wrong K**: Too few or too many clusters
1. **Poor initialization**: Can lead to suboptimal results

## Preprocessing Best Practices

```python
# Standardize features to have mean=0, std=1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Remove outliers before clustering
# Handle missing values
# Consider dimensionality reduction (PCA) for high-dimensional data
```

## Time Complexity

- **Per iteration**: O(n × K × d)
  - n = number of samples
  - K = number of clusters
  - d = number of features
- **Total**: O(n × K × d × i)
  - i = number of iterations

## Space Complexity

- O(n × d) for data storage
- O(K × d) for centroids

## When to Use K-Means

**Good for:**

- Customer segmentation
- Image compression
- Document clustering
- Data preprocessing
- Anomaly detection

**Not ideal for:**

- Non-spherical clusters
- Clusters with very different sizes
- Clusters with different densities
- Data with many outliers

## Comparison with sklearn

```python
from sklearn.cluster import KMeans as SKLearnKMeans

# sklearn implementation
sklearn_kmeans = SKLearnKMeans(n_clusters=3, random_state=42, 
                               init='k-means++', n_init=10)
sklearn_labels = sklearn_kmeans.fit_predict(X)

# Our implementation
our_kmeans = KMeans(n_clusters=3, random_state=42)
our_labels = our_kmeans.fit_predict(X)
```

The sklearn implementation includes additional optimizations and features, but the core algorithm remains the same.

# Neural Networks and Deep Learning

# Natural Language Processing

# Large Language Models

# Reinforcement Learning





# References