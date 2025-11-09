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

# Unsupervised Learning

# Neural Networks and Deep Learning

# Natural Language Processing

# Large Language Models

# Reinforcement Learning





# References