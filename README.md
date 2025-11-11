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

## Prophet-Based approaches

# Meta Prophet Time Series Forecasting Reference Guide

## 📋 Table of Contents

1. [Core Concepts](#core-concepts)
1. [Mathematical Foundation](#mathematical-foundation)
1. [Implementation from Scratch](#implementation-from-scratch)
1. [Prophet Library Usage](#prophet-library-usage)
1. [ML Pipeline Design](#ml-pipeline-design)
1. [Hyperparameter Tuning](#hyperparameter-tuning)
1. [Model Evaluation](#model-evaluation)
1. [Production Deployment](#production-deployment)

-----

## Core Concepts

### What is Prophet?

Prophet is an additive regression model for time series forecasting developed by Meta (Facebook). It’s particularly effective for business time series with:

- Strong seasonal patterns (daily, weekly, yearly)
- Historical trend changes
- Missing data and outliers
- Holiday effects

### When to Use Prophet

✅ **Good for:**

- Business metrics (sales, revenue, engagement)
- Data with clear seasonality
- Missing data points
- Automatic handling of outliers
- Quick prototyping

❌ **Not ideal for:**

- High-frequency data (sub-daily)
- Short time series (<2 cycles)
- Data without clear patterns
- Real-time streaming data

-----

## Mathematical Foundation

### Decomposition Model

Prophet uses an additive model:

```
y(t) = g(t) + s(t) + h(t) + εₜ
```

Where:

- **g(t)**: Trend (piecewise linear or logistic growth)
- **s(t)**: Seasonality (Fourier series)
- **h(t)**: Holiday effects
- **εₜ**: Error term

### 1. Trend Component g(t)

#### Linear Trend (Default)

```
g(t) = (k + a(t)ᵀδ) · t + (m + a(t)ᵀγ)
```

- k: growth rate
- δ: rate adjustments at changepoints
- m: offset parameter
- γ: changepoint adjustments

#### Logistic Growth (Saturating)

```
g(t) = C(t) / (1 + exp(-(k + a(t)ᵀδ)(t - (m + a(t)ᵀγ))))
```

- C(t): carrying capacity

### 2. Seasonality Component s(t)

Uses Fourier series:

```
s(t) = Σ(aₙ · cos(2πnt/P) + bₙ · sin(2πnt/P))
```

- P: period (365.25 for yearly, 7 for weekly)
- N: number of Fourier terms
- Default: N=10 for yearly, N=3 for weekly

### 3. Holiday Component h(t)

```
h(t) = Σ κᵢ · 1(t ∈ Dᵢ)
```

- κᵢ: holiday effect
- Dᵢ: set of days for holiday i

-----

## Implementation from Scratch

### Step 1: Data Preparation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Prophet requires specific column names: 'ds' (date) and 'y' (value)
def prepare_data(df, date_col, value_col):
    """
    Prepare data for Prophet format
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    date_col : str
        Name of date column
    value_col : str
        Name of value column
    
    Returns:
    --------
    DataFrame with 'ds' and 'y' columns
    """
    df_prophet = pd.DataFrame({
        'ds': pd.to_datetime(df[date_col]),
        'y': df[value_col]
    })
    
    # Sort by date
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    
    # Remove duplicates
    df_prophet = df_prophet.drop_duplicates(subset='ds')
    
    return df_prophet

# Example
data = {
    'date': pd.date_range('2020-01-01', periods=365, freq='D'),
    'sales': np.random.randn(365).cumsum() + 100
}
df = pd.DataFrame(data)
df_prepared = prepare_data(df, 'date', 'sales')
```

### Step 2: Building Trend Component

```python
class TrendComponent:
    """Piecewise linear trend with changepoints"""
    
    def __init__(self, n_changepoints=25, changepoint_range=0.8):
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.changepoints = None
        self.k = None  # growth rate
        self.m = None  # offset
        self.delta = None  # rate adjustments
        
    def fit(self, t, y):
        """
        Fit trend to data
        
        Parameters:
        -----------
        t : array
            Time values (days since start)
        y : array
            Target values
        """
        # Set changepoints
        n = len(t)
        s = int(n * self.changepoint_range)
        self.changepoints = np.linspace(0, s, self.n_changepoints + 1)[1:]
        
        # Create design matrix
        A = self._get_changepoint_matrix(t)
        
        # Fit using least squares with regularization
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
        
        # Design matrix: [t, changepoint features]
        X = np.column_stack([t.reshape(-1, 1), A])
        model.fit(X, y)
        
        self.k = model.coef_[0]
        self.delta = model.coef_[1:]
        self.m = model.intercept_
        
    def _get_changepoint_matrix(self, t):
        """Create matrix of changepoint indicators"""
        A = np.zeros((len(t), len(self.changepoints)))
        for i, cp in enumerate(self.changepoints):
            A[:, i] = (t >= cp).astype(float) * (t - cp)
        return A
    
    def predict(self, t):
        """Predict trend values"""
        A = self._get_changepoint_matrix(t)
        return self.k * t + self.m + np.dot(A, self.delta)
```

### Step 3: Building Seasonality Component

```python
class SeasonalityComponent:
    """Fourier series seasonality"""
    
    def __init__(self, period, fourier_order):
        """
        Parameters:
        -----------
        period : float
            Period of seasonality (e.g., 365.25 for yearly, 7 for weekly)
        fourier_order : int
            Number of Fourier terms
        """
        self.period = period
        self.fourier_order = fourier_order
        self.beta = None
        
    def _fourier_series(self, t):
        """Generate Fourier series features"""
        features = []
        for n in range(1, self.fourier_order + 1):
            features.append(np.cos(2 * np.pi * n * t / self.period))
            features.append(np.sin(2 * np.pi * n * t / self.period))
        return np.column_stack(features)
    
    def fit(self, t, y):
        """Fit seasonality to data"""
        from sklearn.linear_model import Ridge
        X = self._fourier_series(t)
        model = Ridge(alpha=0.01)
        model.fit(X, y)
        self.beta = model.coef_
        
    def predict(self, t):
        """Predict seasonal component"""
        X = self._fourier_series(t)
        return np.dot(X, self.beta)
```

### Step 4: Complete Prophet from Scratch

```python
class SimpleProphet:
    """Simplified Prophet implementation"""
    
    def __init__(self, 
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=False,
                 n_changepoints=25):
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.n_changepoints = n_changepoints
        
        self.trend_component = None
        self.seasonality_components = []
        self.start_date = None
        
    def fit(self, df):
        """
        Fit model to data
        
        Parameters:
        -----------
        df : DataFrame
            Must have 'ds' (datetime) and 'y' (value) columns
        """
        # Store start date
        self.start_date = df['ds'].min()
        
        # Convert dates to days since start
        t = (df['ds'] - self.start_date).dt.total_seconds() / (24 * 3600)
        t = t.values
        y = df['y'].values
        
        # 1. Fit trend
        self.trend_component = TrendComponent(n_changepoints=self.n_changepoints)
        self.trend_component.fit(t, y)
        
        # Remove trend
        y_detrended = y - self.trend_component.predict(t)
        
        # 2. Fit seasonality components
        if self.yearly_seasonality:
            yearly = SeasonalityComponent(period=365.25, fourier_order=10)
            yearly.fit(t, y_detrended)
            self.seasonality_components.append(('yearly', yearly))
            y_detrended -= yearly.predict(t)
        
        if self.weekly_seasonality:
            weekly = SeasonalityComponent(period=7, fourier_order=3)
            weekly.fit(t, y_detrended)
            self.seasonality_components.append(('weekly', weekly))
        
    def predict(self, periods):
        """
        Make future predictions
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        
        Returns:
        --------
        DataFrame with predictions
        """
        # Create future dates
        last_date = self.start_date + timedelta(days=len(self.trend_component.changepoints))
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Convert to days since start
        t = (future_dates - self.start_date).total_seconds() / (24 * 3600)
        t = t.values
        
        # Predict components
        yhat = self.trend_component.predict(t)
        
        for name, component in self.seasonality_components:
            yhat += component.predict(t)
        
        # Create result dataframe
        result = pd.DataFrame({
            'ds': future_dates,
            'yhat': yhat
        })
        
        return result
    
    def plot_components(self, df):
        """Plot trend and seasonality components"""
        import matplotlib.pyplot as plt
        
        t = (df['ds'] - self.start_date).dt.total_seconds() / (24 * 3600)
        t = t.values
        
        n_plots = 1 + len(self.seasonality_components)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
        
        # Plot trend
        axes[0].plot(df['ds'], self.trend_component.predict(t))
        axes[0].set_title('Trend')
        axes[0].set_xlabel('Date')
        
        # Plot seasonalities
        for i, (name, component) in enumerate(self.seasonality_components):
            axes[i+1].plot(df['ds'], component.predict(t))
            axes[i+1].set_title(f'{name.capitalize()} Seasonality')
            axes[i+1].set_xlabel('Date')
        
        plt.tight_layout()
        return fig
```

-----

## Prophet Library Usage

### Installation

```bash
pip install prophet
# or for faster stan backend
pip install prophet[stan]
```

### Basic Usage

```python
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# 1. Prepare data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365*3, freq='D'),
    'y': np.random.randn(365*3).cumsum() + 100
})

# 2. Initialize and fit model
model = Prophet(
    growth='linear',  # or 'logistic'
    changepoint_prior_scale=0.05,  # flexibility of trend (0.001-0.5)
    seasonality_prior_scale=10,     # flexibility of seasonality
    seasonality_mode='additive',    # or 'multiplicative'
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(df)

# 3. Make predictions
future = model.make_future_dataframe(periods=365)  # forecast 1 year
forecast = model.predict(future)

# 4. Visualize
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
plt.show()
```

### Advanced Features

#### Adding Custom Seasonality

```python
# Monthly seasonality
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# Conditional seasonality (e.g., only on weekends)
df['on_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)

model.add_seasonality(
    name='weekend_seasonality',
    period=7,
    fourier_order=3,
    condition_name='on_weekend'
)

forecast = model.predict(future)
```

#### Adding Regressors

```python
# Add external variables
df['temperature'] = np.random.randn(len(df)) * 10 + 20
df['marketing_spend'] = np.random.exponential(1000, len(df))

model = Prophet()
model.add_regressor('temperature')
model.add_regressor('marketing_spend', mode='multiplicative')

model.fit(df)

# Future dataframe must include regressors
future = model.make_future_dataframe(periods=30)
future['temperature'] = np.random.randn(len(future)) * 10 + 20
future['marketing_spend'] = np.random.exponential(1000, len(future))

forecast = model.predict(future)
```

#### Handling Holidays

```python
# Define holidays
holidays = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2020-12-25', '2021-12-25', '2022-12-25']),
    'lower_window': -2,  # 2 days before
    'upper_window': 1,   # 1 day after
})

# Built-in country holidays
from prophet.make_holidays import make_holidays_df
us_holidays = make_holidays_df(
    year_list=[2020, 2021, 2022],
    country='US'
)

model = Prophet(holidays=holidays)
# or
model = Prophet(holidays=us_holidays)

model.fit(df)
```

#### Saturation and Capacity

```python
# For logistic growth
df['cap'] = 1000  # maximum capacity
df['floor'] = 0   # minimum capacity (optional)

model = Prophet(growth='logistic')
model.fit(df)

future = model.make_future_dataframe(periods=365)
future['cap'] = 1000
future['floor'] = 0

forecast = model.predict(future)
```

-----

## ML Pipeline Design

### Complete Pipeline Architecture

```python
class ProphetMLPipeline:
    """Production-ready Prophet ML pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.metrics = {}
        
    def load_data(self, filepath):
        """Load and validate data"""
        df = pd.read_csv(filepath)
        
        # Validate required columns
        assert 'ds' in df.columns, "Missing 'ds' column"
        assert 'y' in df.columns, "Missing 'y' column"
        
        # Convert types
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        return df
    
    def preprocess(self, df):
        """Data preprocessing"""
        df = df.copy()
        
        # 1. Handle missing values
        df = df.dropna(subset=['ds', 'y'])
        
        # 2. Remove outliers (optional)
        if self.config.get('remove_outliers', False):
            q1 = df['y'].quantile(0.01)
            q99 = df['y'].quantile(0.99)
            df = df[(df['y'] >= q1) & (df['y'] <= q99)]
        
        # 3. Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        # 4. Handle duplicates
        df = df.drop_duplicates(subset='ds', keep='first')
        
        # 5. Fill gaps (optional)
        if self.config.get('fill_gaps', False):
            df = df.set_index('ds').asfreq('D').reset_index()
            df['y'] = df['y'].interpolate(method='linear')
        
        return df
    
    def feature_engineering(self, df):
        """Create additional features"""
        df = df.copy()
        
        # Time-based features
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in self.config.get('lags', []):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling statistics
        for window in self.config.get('rolling_windows', []):
            df[f'rolling_mean_{window}'] = df['y'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['y'].rolling(window).std()
        
        return df
    
    def train_test_split(self, df, test_size=0.2):
        """Time-based train/test split"""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        return train, test
    
    def build_model(self):
        """Initialize Prophet model with config"""
        model_params = {
            'growth': self.config.get('growth', 'linear'),
            'changepoint_prior_scale': self.config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': self.config.get('seasonality_prior_scale', 10),
            'seasonality_mode': self.config.get('seasonality_mode', 'additive'),
            'yearly_seasonality': self.config.get('yearly_seasonality', True),
            'weekly_seasonality': self.config.get('weekly_seasonality', True),
            'daily_seasonality': self.config.get('daily_seasonality', False),
            'interval_width': self.config.get('interval_width', 0.80)
        }
        
        self.model = Prophet(**model_params)
        
        # Add custom seasonalities
        for seasonality in self.config.get('custom_seasonalities', []):
            self.model.add_seasonality(**seasonality)
        
        # Add regressors
        for regressor in self.config.get('regressors', []):
            self.model.add_regressor(regressor)
        
        return self.model
    
    def train(self, train_df):
        """Train the model"""
        print("Training Prophet model...")
        self.model.fit(train_df)
        print("Training complete!")
        
    def predict(self, periods=None, future_df=None):
        """Make predictions"""
        if future_df is None:
            future_df = self.model.make_future_dataframe(periods=periods)
        
        forecast = self.model.predict(future_df)
        return forecast
    
    def evaluate(self, test_df):
        """Evaluate model performance"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Predict on test set
        forecast = self.model.predict(test_df[['ds']])
        
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        # Calculate metrics
        self.metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred)
        }
        
        return self.metrics
    
    def cross_validate_prophet(self, df, horizon='30 days', initial='730 days', period='180 days'):
        """Time series cross-validation"""
        from prophet.diagnostics import cross_validation, performance_metrics
        
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        metrics = performance_metrics(df_cv)
        return df_cv, metrics
    
    def save_model(self, filepath):
        """Save trained model"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        """Load trained model"""
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

# Example usage
config = {
    'growth': 'linear',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10,
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'remove_outliers': True,
    'fill_gaps': True
}

pipeline = ProphetMLPipeline(config)
df = pipeline.load_data('data.csv')
df = pipeline.preprocess(df)
train, test = pipeline.train_test_split(df)

pipeline.build_model()
pipeline.train(train)
metrics = pipeline.evaluate(test)
print(metrics)
```

-----

## Hyperparameter Tuning

### Grid Search for Prophet

```python
from itertools import product
import pandas as pd

def prophet_grid_search(df, param_grid, metric='mape'):
    """
    Perform grid search for Prophet hyperparameters
    
    Parameters:
    -----------
    df : DataFrame
        Training data
    param_grid : dict
        Dictionary of parameters to search
    metric : str
        Metric to optimize ('mape', 'rmse', 'mae')
    
    Returns:
    --------
    best_params : dict
        Best parameter combination
    results : DataFrame
        All results
    """
    from prophet.diagnostics import cross_validation, performance_metrics
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    results = []
    
    for i, params in enumerate(combinations):
        print(f"Testing combination {i+1}/{len(combinations)}: {params}")
        
        try:
            # Build model
            model = Prophet(**params)
            model.fit(df)
            
            # Cross-validate
            df_cv = cross_validation(
                model,
                initial='730 days',
                period='180 days',
                horizon='30 days'
            )
            
            # Calculate metrics
            df_metrics = performance_metrics(df_cv)
            
            # Store results
            result = params.copy()
            result['mape'] = df_metrics['mape'].mean()
            result['rmse'] = df_metrics['rmse'].mean()
            result['mae'] = df_metrics['mae'].mean()
            results.append(result)
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    best_idx = results_df[metric].idxmin()
    best_params = results_df.loc[best_idx].to_dict()
    
    return best_params, results_df

# Example usage
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

best_params, results = prophet_grid_search(df, param_grid)
print("Best parameters:", best_params)
```

### Bayesian Optimization

```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

def prophet_objective(params):
    """Objective function for hyperopt"""
    
    # Build model with params
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        seasonality_mode=params['seasonality_mode']
    )
    
    model.fit(train_df)
    
    # Evaluate
    forecast = model.predict(test_df[['ds']])
    mape = np.mean(np.abs((test_df['y'] - forecast['yhat']) / test_df['y'])) * 100
    
    return {'loss': mape, 'status': STATUS_OK}

# Define search space
space = {
    'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
    'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10)),
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
}

# Run optimization
trials = Trials()
best = fmin(
    fn=prophet_objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

print("Best parameters:", best)
```

-----

## Model Evaluation

### Comprehensive Evaluation Metrics

```python
def evaluate_forecast(y_true, y_pred, y_pred_lower=None, y_pred_upper=None):
    """
    Comprehensive forecast evaluation
    
    Returns dict with multiple metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {}
    
    # Basic metrics
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # Percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['MAPE'] = mape
    
    # Symmetric MAPE (better for values near zero)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    metrics['SMAPE'] = smape
    
    # Weighted metrics
    metrics['WMAPE'] = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    # Direction accuracy
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        direction_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
        metrics['Direction_Accuracy'] = direction_accuracy
    
    # Coverage (if prediction intervals provided)
    if y_pred_lower is not None and y_pred_upper is not None:
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper)) * 100
        metrics['Coverage'] = coverage
    
    return metrics

# Usage
forecast = model.predict(test_df[['ds']])
metrics = evaluate_forecast(
    y_true=test_df['y'].values,
    y_pred=forecast['yhat'].values,
    y_pred_lower=forecast['yhat_lower'].values,
    y_pred_upper=forecast['yhat_upper'].values
)

for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")
```

### Visualization Functions

```python
def plot_forecast_evaluation(df_train, df_test, forecast, model_name='Prophet'):
    """Comprehensive forecast visualization"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Actual vs Predicted
    ax = axes[0, 0]
    ax.plot(df_train['ds'], df_train['y'], 'k.', label='Training', alpha=0.5)
    ax.plot(df_test['ds'], df_test['y'], 'b.', label='Actual Test', markersize=8)
    ax.plot(forecast['ds'], forecast['yhat'], 'r-', label='Forecast', linewidth=2)
    ax.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'],
                     alpha=0.3, color='red', label='80% Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(f'{model_name}: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals over time
    ax = axes[0, 1]
    residuals = df_test['y'].values - forecast['yhat'].values
    ax.scatter(forecast['ds'], residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Over Time')
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Residuals Distribution')
    ax.grid(True, alpha=0.3)
    
    # 4. Actual vs Predicted scatter
    ax = axes[1, 1]
    ax.scatter(df_test['y'].values, forecast['yhat'].values, alpha=0.6)
    min_val = min(df_test['y'].min(), forecast['yhat'].min())
    max_val = max(df_test['y'].max(), forecast['yhat'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Scatter')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Usage
fig = plot_forecast_evaluation(train_df, test_df, forecast)

def plot_diagnostics(model, df):
    """Plot Prophet diagnostic plots"""
    import matplotlib.pyplot as plt
    from prophet.plot import plot_cross_validation_metric
    from prophet.diagnostics import cross_validation, performance_metrics
    
    # Cross-validation
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
    df_p = performance_metrics(df_cv)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot metrics over horizon
    metrics = ['mape', 'rmse', 'mae', 'coverage']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        plot_cross_validation_metric(df_cv, metric=metric, ax=ax)
        ax.set_title(f'{metric.upper()} over Forecast Horizon')
    
    plt.tight_layout()
    return fig, df_cv, df_p
```

-----

## Production Deployment

### Model Serialization and Versioning

```python
import pickle
import json
from datetime import datetime
import hashlib

class ModelRegistry:
    """Manage model versions and metadata"""
    
    def __init__(self, registry_path='models/'):
        self.registry_path = registry_path
        import os
        os.makedirs(registry_path, exist_ok=True)
        
    def save_model(self, model, metadata, version=None):
        """
        Save model with metadata
        
        Parameters:
        -----------
        model : Prophet
            Trained model
        metadata : dict
            Model metadata (metrics, params, etc.)
        version : str
            Model version (auto-generated if None)
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f"{self.registry_path}/model_{version}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        
        # Save metadata
        metadata['version'] = version
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['model_hash'] = model_hash
        metadata['model_path'] = model_path
        
        metadata_path = f"{self.registry_path}/metadata_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return version
    
    def load_model(self, version):
        """Load model by version"""
        model_path = f"{self.registry_path}/model_{version}.pkl"
        metadata_path = f"{self.registry_path}/metadata_{version}.json"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def list_models(self):
        """List all saved models"""
        import os
        import glob
        
        metadata_files = glob.glob(f"{self.registry_path}/metadata_*.json")
        models = []
        
        for mf in metadata_files:
            with open(mf, 'r') as f:
                metadata = json.load(f)
                models.append(metadata)
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)

# Usage
registry = ModelRegistry()

# Save model
metadata = {
    'model_type': 'Prophet',
    'metrics': {'MAPE': 5.2, 'RMSE': 10.5},
    'params': {'changepoint_prior_scale': 0.05},
    'training_samples': len(train_df),
    'features': ['ds', 'y']
}

version = registry.save_model(model, metadata)

# Load model
loaded_model, loaded_metadata = registry.load_model(version)

# List all models
all_models = registry.list_models()
print(f"Found {len(all_models)} models")
```

### REST API with Flask

```python
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import pickle

app = Flask(__name__)

# Load model at startup
with open('models/prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request body:
    {
        "periods": 30,
        "freq": "D"
    }
    
    or
    
    {
        "dates": ["2024-01-01", "2024-01-02", ...]
    }
    """
    try:
        data = request.get_json()
        
        # Create future dataframe
        if 'dates' in data:
            future = pd.DataFrame({'ds': pd.to_datetime(data['dates'])})
        else:
            periods = data.get('periods', 30)
            freq = data.get('freq', 'D')
            future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make prediction
        forecast = model.predict(future)
        
        # Format response
        result = {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'),
            'metadata': {
                'model_version': '1.0',
                'prediction_timestamp': datetime.now().isoformat(),
                'num_predictions': len(forecast)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Request body:
    {
        "data": [
            {"ds": "2024-01-01"},
            {"ds": "2024-01-02"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        df['ds'] = pd.to_datetime(df['ds'])
        
        forecast = model.predict(df)
        
        result = {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/components', methods=['POST'])
def get_components():
    """
    Get forecast components (trend, seasonality, etc.)
    """
    try:
        data = request.get_json()
        periods = data.get('periods', 30)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        components = {
            'trend': forecast[['ds', 'trend']].to_dict(orient='records'),
            'yearly': forecast[['ds', 'yearly']].to_dict(orient='records') if 'yearly' in forecast.columns else None,
            'weekly': forecast[['ds', 'weekly']].to_dict(orient='records') if 'weekly' in forecast.columns else None,
        }
        
        return jsonify(components)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  prophet-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/prophet_model.pkl
    restart: unless-stopped
```

### Monitoring and Logging

```python
import logging
from functools import wraps
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prophet_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_prediction(func):
    """Decorator to log predictions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            logger.info(f"Prediction successful - Duration: {duration:.3f}s")
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise
    
    return wrapper

class PredictionMonitor:
    """Monitor prediction performance"""
    
    def __init__(self):
        self.predictions = []
        
    def record(self, y_true, y_pred, timestamp=None):
        """Record a prediction for monitoring"""
        if timestamp is None:
            timestamp = datetime.now()
        
        error = abs(y_true - y_pred)
        pct_error = (error / y_true * 100) if y_true != 0 else 0
        
        self.predictions.append({
            'timestamp': timestamp,
            'y_true': y_true,
            'y_pred': y_pred,
            'error': error,
            'pct_error': pct_error
        })
    
    def get_stats(self, window='24h'):
        """Get statistics for recent predictions"""
        if not self.predictions:
            return {}
        
        df = pd.DataFrame(self.predictions)
        
        # Filter by time window
        if window:
            cutoff = datetime.now() - pd.Timedelta(window)
            df = df[df['timestamp'] > cutoff]
        
        if len(df) == 0:
            return {}
        
        return {
            'count': len(df),
            'mean_error': df['error'].mean(),
            'mean_pct_error': df['pct_error'].mean(),
            'max_error': df['error'].max(),
            'p95_error': df['error'].quantile(0.95)
        }
    
    def check_drift(self, threshold=10.0):
        """Check if model performance is drifting"""
        stats = self.get_stats(window='24h')
        
        if not stats:
            return False
        
        if stats['mean_pct_error'] > threshold:
            logger.warning(f"Model drift detected! Mean error: {stats['mean_pct_error']:.2f}%")
            return True
        
        return False

# Usage
monitor = PredictionMonitor()

@log_prediction
def make_prediction(data):
    forecast = model.predict(data)
    return forecast

# Record actual vs predicted for monitoring
monitor.record(y_true=100, y_pred=95)
stats = monitor.get_stats()
print(stats)
```

-----

## Best Practices & Tips

### 1. Data Quality Checklist

```python
def validate_prophet_data(df):
    """
    Validate data before training
    
    Returns: dict with validation results
    """
    validation = {
        'passed': True,
        'warnings': [],
        'errors': []
    }
    
    # Check required columns
    if 'ds' not in df.columns:
        validation['errors'].append("Missing 'ds' column")
        validation['passed'] = False
    
    if 'y' not in df.columns:
        validation['errors'].append("Missing 'y' column")
        validation['passed'] = False
    
    if not validation['passed']:
        return validation
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['ds']):
        validation['errors'].append("'ds' must be datetime type")
        validation['passed'] = False
    
    if not pd.api.types.is_numeric_dtype(df['y']):
        validation['errors'].append("'y' must be numeric type")
        validation['passed'] = False
    
    # Check for missing values
    missing_ds = df['ds'].isna().sum()
    missing_y = df['y'].isna().sum()
    
    if missing_ds > 0:
        validation['warnings'].append(f"{missing_ds} missing dates")
    
    if missing_y > 0:
        validation['warnings'].append(f"{missing_y} missing values")
    
    # Check for duplicates
    duplicates = df['ds'].duplicated().sum()
    if duplicates > 0:
        validation['warnings'].append(f"{duplicates} duplicate dates")
    
    # Check data length
    if len(df) < 20:
        validation['warnings'].append(f"Only {len(df)} data points (recommend 100+)")
    
    # Check for negative values
    if (df['y'] < 0).any():
        validation['warnings'].append("Negative values present (consider floor=0)")
    
    # Check for outliers
    q1 = df['y'].quantile(0.25)
    q3 = df['y'].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df['y'] < (q1 - 3 * iqr)) | (df['y'] > (q3 + 3 * iqr))).sum()
    
    if outliers > 0:
        validation['warnings'].append(f"{outliers} potential outliers detected")
    
    return validation

# Usage
validation = validate_prophet_data(df)
if not validation['passed']:
    print("Validation failed:", validation['errors'])
else:
    print("Validation passed")
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
```

### 2. Performance Optimization

```python
# Use cmdstanpy for faster inference (5-10x speedup)
# pip install prophet[stan]

# Parallel cross-validation
from prophet.diagnostics import cross_validation

df_cv = cross_validation(
    model,
    initial='730 days',
    period='180 days',
    horizon='30 days',
    parallel='processes'  # Use multiprocessing
)

# Reduce MCMC samples for faster training
model = Prophet(
    mcmc_samples=100,  # Default is 0 (use MAP)
    uncertainty_samples=100  # Reduce from default 1000
)

# Disable unnecessary seasonalities
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=False,  # If not needed
    yearly_seasonality='auto'  # Auto-detect
)
```

### 3. Common Pitfalls and Solutions

```python
"""
PITFALL 1: Not enough data
Solution: Need at least 2 complete seasonal cycles
"""
# For monthly data with yearly seasonality: 24+ months
# For daily data with weekly seasonality: 14+ days

"""
PITFALL 2: Irregular time intervals
Solution: Standardize to regular frequency
"""
# Resample to daily frequency
df = df.set_index('ds').resample('D').asfreq()
df['y'] = df['y'].interpolate()

"""
PITFALL 3: Not handling capacity for logistic growth
Solution: Always set cap when using logistic growth
"""
df['cap'] = df['y'].max() * 1.5  # Set capacity
model = Prophet(growth='logistic')

"""
PITFALL 4: Ignoring domain knowledge
Solution: Add custom seasonalities and regressors
"""
# Add payday effect
df['is_payday'] = df['ds'].dt.day.isin([1, 15]).astype(int)
model.add_regressor('is_payday')

"""
PITFALL 5: Over-fitting with too many changepoints
Solution: Use regularization
"""
model = Prophet(
    changepoint_prior_scale=0.05,  # Lower = less flexible
    n_changepoints=25  # Fewer changepoints
)

"""
PITFALL 6: Not validating on hold-out set
Solution: Always do time-series split validation
"""
# Never shuffle! Use time-based split
split_date = df['ds'].max() - pd.Timedelta(days=90)
train = df[df['ds'] <= split_date]
test = df[df['ds'] > split_date]
```

### 4. Interpretability and Explainability

```python
def explain_forecast(model, forecast, date_to_explain):
    """
    Break down forecast into interpretable components
    
    Parameters:
    -----------
    model : Prophet
        Fitted model
    forecast : DataFrame
        Forecast output
    date_to_explain : datetime
        Date to explain
    
    Returns:
    --------
    dict with component contributions
    """
    # Get forecast for specific date
    row = forecast[forecast['ds'] == date_to_explain].iloc[0]
    
    explanation = {
        'date': date_to_explain,
        'forecast': row['yhat'],
        'components': {}
    }
    
    # Trend
    if 'trend' in row:
        explanation['components']['trend'] = row['trend']
    
    # Seasonalities
    for col in forecast.columns:
        if col.endswith('_seasonal') or col in ['yearly', 'weekly', 'daily']:
            explanation['components'][col] = row[col]
    
    # Holidays
    if 'holidays' in row:
        explanation['components']['holidays'] = row['holidays']
    
    # Additional regressors
    for col in forecast.columns:
        if col.startswith('extra_regressors'):
            explanation['components'][col] = row[col]
    
    # Calculate percentages
    total = sum(explanation['components'].values())
    if total != 0:
        explanation['percentages'] = {
            k: (v / total * 100) for k, v in explanation['components'].items()
        }
    
    return explanation

# Usage
from datetime import datetime
explanation = explain_forecast(
    model, 
    forecast, 
    datetime(2024, 12, 25)
)

print(f"Forecast for {explanation['date']}: {explanation['forecast']:.2f}")
print("\nComponent contributions:")
for component, value in explanation['components'].items():
    pct = explanation['percentages'][component]
    print(f"  {component}: {value:.2f} ({pct:.1f}%)")
```

### 5. A/B Testing Models

```python
def compare_models(models_dict, test_df):
    """
    Compare multiple Prophet models
    
    Parameters:
    -----------
    models_dict : dict
        {'model_name': model_object}
    test_df : DataFrame
        Test data
    
    Returns:
    --------
    DataFrame with comparison results
    """
    results = []
    
    for name, model in models_dict.items():
        forecast = model.predict(test_df[['ds']])
        
        metrics = evaluate_forecast(
            test_df['y'].values,
            forecast['yhat'].values,
            forecast['yhat_lower'].values,
            forecast['yhat_upper'].values
        )
        
        metrics['model'] = name
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAPE')
    
    return results_df

# Usage
models = {
    'baseline': Prophet(),
    'with_regressors': Prophet().add_regressor('temperature'),
    'multiplicative': Prophet(seasonality_mode='multiplicative'),
    'tuned': Prophet(changepoint_prior_scale=0.01)
}

# Train all models
for name, model in models.items():
    model.fit(train_df)

# Compare
comparison = compare_models(models, test_df)
print(comparison)
```

-----

## Quick Reference Commands

### Installation

```bash
# Standard installation
pip install prophet

# With Stan backend (faster)
pip install prophet[stan]

# Development version
pip install git+https://github.com/facebook/prophet.git
```

### Essential Imports

```python
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import pandas as pd
import numpy as np
```

### Minimal Working Example

```python
# 1. Prepare data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365*2, freq='D'),
    'y': np.random.randn(365*2).cumsum() + 100
})

# 2. Train
m = Prophet()
m.fit(df)

# 3. Predict
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# 4. Plot
fig = m.plot(forecast)
fig_comp = m.plot_components(forecast)
```

### Common Patterns

```python
# Custom seasonality
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# External regressor
m.add_regressor('temperature')

# Holidays
from prophet.make_holidays import make_holidays_df
holidays = make_holidays_df(year_list=[2020, 2021, 2022], country='US')
m = Prophet(holidays=holidays)

# Logistic growth
df['cap'] = 1000
m = Prophet(growth='logistic')

# Cross-validation
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='30 days')
df_p = performance_metrics(df_cv)
```

-----

## Resources and Further Reading

### Official Documentation

- Prophet Documentation: https://facebook.github.io/prophet/
- API Reference: https://facebook.github.io/prophet/docs/
- GitHub Repository: https://github.com/facebook/prophet

### Academic Papers

- Taylor, S.J., Letham, B. (2018). “Forecasting at Scale.” The American Statistician 72(1):37-45

### Key Parameters Summary

|Parameter              |Default   |Range                       |Description                           |
|-----------------------|----------|----------------------------|--------------------------------------|
|growth                 |‘linear’  |‘linear’, ‘logistic’, ‘flat’|Type of trend                         |
|changepoints           |None      |list of dates               |Manual changepoints                   |
|n_changepoints         |25        |int                         |Number of automatic changepoints      |
|changepoint_range      |0.8       |0-1                         |Proportion of history for changepoints|
|changepoint_prior_scale|0.05      |0.001-0.5                   |Flexibility of trend changes          |
|seasonality_prior_scale|10        |0.01-10                     |Flexibility of seasonality            |
|seasonality_mode       |‘additive’|‘additive’, ‘multiplicative’|Type of seasonality                   |
|yearly_seasonality     |‘auto’    |True, False, int            |Yearly seasonality                    |
|weekly_seasonality     |‘auto’    |True, False, int            |Weekly seasonality                    |
|daily_seasonality      |‘auto’    |True, False, int            |Daily seasonality                     |
|interval_width         |0.80      |0-1                         |Width of uncertainty intervals        |

### Troubleshooting Guide

**Problem:** Model too rigid (under-fitting)

- ✅ Increase `changepoint_prior_scale`
- ✅ Increase Fourier order for seasonality
- ✅ Add more changepoints

**Problem:** Model too flexible (over-fitting)

- ✅ Decrease `changepoint_prior_scale`
- ✅ Decrease Fourier order
- ✅ Use cross-validation to tune

**Problem:** Poor uncertainty intervals

- ✅ Increase `uncertainty_samples`
- ✅ Check for outliers in training data
- ✅ Use MCMC: `mcmc_samples=300`

**Problem:** Slow training

- ✅ Reduce data size (aggregate if possible)
- ✅ Decrease Fourier orders
- ✅ Install cmdstanpy backend
- ✅ Use parallel cross-validation

**Problem:** Negative forecasts

- ✅ Set floor: `df['floor'] = 0`
- ✅ Use logistic growth with cap
- ✅ Apply log transformation


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

## Perceptron

# Perceptron & Deep Learning Implementation Guide

## Table of Contents

1. [Perceptron Fundamentals](#perceptron-fundamentals)
1. [Implementation from Scratch](#implementation-from-scratch)
1. [Deep Learning Process Design](#deep-learning-process-design)
1. [Advanced Concepts](#advanced-concepts)

-----

## Perceptron Fundamentals

### What is a Perceptron?

A perceptron is the simplest neural network unit - a binary classifier that learns a linear decision boundary.

**Mathematical Model:**

```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias)
      = activation(w·x + b)
```

**Key Components:**

- **Weights (w)**: Learn the importance of each feature
- **Bias (b)**: Shifts the decision boundary
- **Activation Function**: Introduces non-linearity (step, sigmoid, ReLU, etc.)
- **Learning Rate (α)**: Controls how fast weights update

-----

## Implementation from Scratch

### 1. Single Perceptron (Binary Classification)

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize perceptron
        
        Parameters:
        - learning_rate: Step size for weight updates
        - n_iterations: Number of training epochs
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def activation(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        """
        Train the perceptron
        
        Parameters:
        - X: Training data (n_samples, n_features)
        - y: Target labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for iteration in range(self.n_iterations):
            epoch_loss = 0
            
            for idx, x_i in enumerate(X):
                # Forward pass
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Calculate error
                error = y[idx] - y_predicted
                epoch_loss += abs(error)
                
                # Update weights and bias (Perceptron Learning Rule)
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error
            
            self.losses.append(epoch_loss)
            
            # Early stopping if converged
            if epoch_loss == 0:
                print(f"Converged at iteration {iteration}")
                break
    
    def predict(self, X):
        """Make predictions on new data"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# Example Usage
if __name__ == "__main__":
    # Create simple linearly separable dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                n_informative=2, n_clusters_per_class=1, 
                                flip_y=0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X_train, y_train)
    
    # Evaluate
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
```

### 2. Multi-Layer Perceptron (Deep Learning)

```python
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize Multi-Layer Perceptron
        
        Parameters:
        - layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        - learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.losses = []
        
        # Initialize weights and biases using He initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network
        
        Returns: List of activations for each layer
        """
        activations = [X]
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Apply activation (ReLU for hidden, sigmoid for output)
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            
            activations.append(a)
        
        return activations
    
    def backward_propagation(self, X, y, activations):
        """
        Backward pass - compute gradients
        
        Returns: Gradients for weights and biases
        """
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        delta = activations[-1] - y.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.relu_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gradients_w[i]
            self.biases[i] -= self.lr * gradients_b[i]
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Parameters:
        - X: Training data
        - y: Target labels
        - epochs: Number of training iterations
        - batch_size: Mini-batch size for training
        - verbose: Print progress
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                activations = self.forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, activations[-1])
                epoch_loss += batch_loss
                
                # Backward pass
                gradients_w, gradients_b = self.backward_propagation(X_batch, y_batch, activations)
                
                # Update parameters
                self.update_parameters(gradients_w, gradients_b)
            
            # Record average loss
            avg_loss = epoch_loss / (n_samples / batch_size)
            self.losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        activations = self.forward_propagation(X)
        return (activations[-1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        activations = self.forward_propagation(X)
        return activations[-1]

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    # Create non-linearly separable dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train neural network
    nn = NeuralNetwork(layer_sizes=[2, 16, 8, 1], learning_rate=0.1)
    nn.fit(X_train, y_train, epochs=500, batch_size=32)
    
    # Evaluate
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions.flatten() == y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
```

-----

## Deep Learning Process Design

### Complete ML Pipeline Workflow

```python
class MLPipeline:
    """
    Complete machine learning pipeline for deep learning projects
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = {}
    
    # STEP 1: Data Preparation
    def load_and_explore_data(self, data_source):
        """
        Load data and perform exploratory analysis
        """
        import pandas as pd
        
        # Load data
        if isinstance(data_source, str):
            df = pd.read_csv(data_source)
        else:
            df = data_source
        
        print("Dataset Shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nBasic statistics:")
        print(df.describe())
        
        return df
    
    # STEP 2: Data Preprocessing
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess features: handle missing values, scale, encode
        """
        from sklearn.preprocessing import StandardScaler
        
        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        # Feature scaling
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    # STEP 3: Feature Engineering
    def engineer_features(self, X):
        """
        Create new features from existing ones
        """
        # Example: polynomial features, interactions, etc.
        X_engineered = X.copy()
        
        # Add squared features
        n_features = X.shape[1]
        X_squared = X ** 2
        X_engineered = np.hstack([X_engineered, X_squared])
        
        return X_engineered
    
    # STEP 4: Train-Validation-Test Split
    def split_data(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Split data into train, validation, and test sets
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train + (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1-train_size), random_state=42
        )
        
        # Second split: val + test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # STEP 5: Model Training with Validation
    def train_model(self, X_train, y_train, X_val, y_val, 
                   layer_sizes, learning_rate=0.01, epochs=1000):
        """
        Train model with validation monitoring
        """
        self.model = NeuralNetwork(layer_sizes, learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train on training data
            activations = self.model.forward_propagation(X_train)
            train_loss = self.model.compute_loss(y_train, activations[-1])
            gradients_w, gradients_b = self.model.backward_propagation(
                X_train, y_train, activations
            )
            self.model.update_parameters(gradients_w, gradients_b)
            
            # Validate
            val_activations = self.model.forward_propagation(X_val)
            val_loss = self.model.compute_loss(y_val, val_activations[-1])
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        return train_losses, val_losses
    
    # STEP 6: Model Evaluation
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = np.mean(predictions.flatten() == y_test)
        
        # Confusion matrix
        tp = np.sum((predictions.flatten() == 1) & (y_test == 1))
        tn = np.sum((predictions.flatten() == 0) & (y_test == 0))
        fp = np.sum((predictions.flatten() == 1) & (y_test == 0))
        fn = np.sum((predictions.flatten() == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': [[tn, fp], [fn, tp]]
        }
        
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        return self.metrics
    
    # STEP 7: Hyperparameter Tuning
    def grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        """
        Simple grid search for hyperparameter tuning
        """
        best_params = None
        best_val_loss = float('inf')
        results = []
        
        for lr in param_grid['learning_rate']:
            for layers in param_grid['layer_sizes']:
                print(f"\nTrying: lr={lr}, layers={layers}")
                
                model = NeuralNetwork(layers, lr)
                
                # Quick training
                for _ in range(100):
                    activations = model.forward_propagation(X_train)
                    gradients_w, gradients_b = model.backward_propagation(
                        X_train, y_train, activations
                    )
                    model.update_parameters(gradients_w, gradients_b)
                
                # Evaluate
                val_activations = model.forward_propagation(X_val)
                val_loss = model.compute_loss(y_val, val_activations[-1])
                
                results.append({
                    'learning_rate': lr,
                    'layer_sizes': layers,
                    'val_loss': val_loss
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {'learning_rate': lr, 'layer_sizes': layers}
        
        print(f"\nBest parameters: {best_params}")
        return best_params, results
```

-----

## Advanced Concepts

### 1. Activation Functions

```python
class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### 2. Regularization Techniques

```python
class Regularization:
    @staticmethod
    def l1_regularization(weights, lambda_param):
        """L1 regularization (Lasso)"""
        return lambda_param * np.sum([np.sum(np.abs(w)) for w in weights])
    
    @staticmethod
    def l2_regularization(weights, lambda_param):
        """L2 regularization (Ridge)"""
        return lambda_param * np.sum([np.sum(w ** 2) for w in weights])
    
    @staticmethod
    def dropout(x, dropout_rate=0.5, training=True):
        """Dropout regularization"""
        if training:
            mask = np.random.binomial(1, 1-dropout_rate, size=x.shape) / (1-dropout_rate)
            return x * mask
        return x
```

### 3. Optimization Algorithms

```python
class Optimizers:
    class SGD:
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.lr = learning_rate
            self.momentum = momentum
            self.velocity = None
        
        def update(self, weights, gradients):
            if self.velocity is None:
                self.velocity = [np.zeros_like(w) for w in weights]
            
            for i in range(len(weights)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * gradients[i]
                weights[i] += self.velocity[i]
            
            return weights
    
    class Adam:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.lr = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = None
            self.v = None
            self.t = 0
        
        def update(self, weights, gradients):
            if self.m is None:
                self.m = [np.zeros_like(w) for w in weights]
                self.v = [np.zeros_like(w) for w in weights]
            
            self.t += 1
            
            for i in range(len(weights)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            return weights
```

### 4. Batch Normalization

```python
class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
    
    def forward(self, x, training=True):
        if self.gamma is None:
            self.gamma = np.ones(x.shape[1])
            self.beta = np.zeros(x.shape[1])
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.ones(x.shape[1])
        
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta
```

-----

## Key Formulas Reference

### Perceptron Learning Rule

```
Δw = α × (y_true - y_pred) × x
Δb = α × (y_true - y_pred)
```

### Gradient Descent

```
w_new = w_old - α × ∂L/∂w
```

### Backpropagation

```
δ_L = (a_L - y) ⊙ σ'(z_L)
δ_l = (W_{l+1}^T δ_{l+1}) ⊙ σ'(z_l)
∂L/∂W_l = δ_l × a_{l-1}^T
```

### Common Loss Functions

- **Binary Cross-Entropy**: `-[y log(ŷ) + (1-y) log(1-ŷ)]`
- **Mean Squared Error**: `(1/n) Σ(y - ŷ)²`
- **Categorical Cross-Entropy**: `-Σ y_i log(ŷ_i)`

-----

## Best Practices Checklist

✅ **Data Preparation**

- Handle missing values
- Scale/normalize features
- Split data properly (train/val/test)
- Check for data leakage

✅ **Model Design**

- Start simple, increase complexity gradually
- Use appropriate activation functions
- Initialize weights properly (Xavier/He)
- Add regularization if overfitting

✅ **Training**

- Monitor both training and validation loss
- Use early stopping
- Try different learning rates
- Use mini-batch training

✅ **Evaluation**

- Use multiple metrics (accuracy, precision, recall, F1)
- Analyze confusion matrix
- Test on unseen data
- Check for bias/fairness

✅ **Optimization**

- Tune hyperparameters systematically
- Use cross-validation
- Try different optimizers
- Consider ensemble methods

-----

## Common Pitfalls to Avoid

❌ Training on entire dataset without validation split
❌ Not scaling features
❌ Using too high learning rate (divergence)
❌ Not monitoring for overfitting
❌ Ignoring class imbalance
❌ Testing on training data
❌ Poor weight initialization
❌ Not shuffling data between epochs

# Natural Language Processing

# Large Language Models

# Reinforcement Learning





# References