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

## Support Vector Machines

## Decision Trees

## Ensemble Learning and Random Forests

# Unsupervised Learning

# Neural Networks and Deep Learning

# Natural Language Processing

# Large Language Models

# Reinforcement Learning





# References