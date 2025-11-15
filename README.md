# MAGE: Multi-Agent Genetic Evolution Neural Framework

## Overview

MAGE is a fully evolutionary neural architecture search (NAS) + neural network trainer that evolves:

* Hidden layer sizes
* Activation functions
* Weights and biases
* Hyperparameters

It supports **three tasks**:

* **Regression** (MSE loss)
* **Binary Classification** (Binary Cross-Entropy)
* **Multiclass Classification** (Softmax Cross-Entropy)

You define the input/output size, initial architecture, and training settings. MAGE then evolves multiple neural agents and selects the fittest ones.

---

# 1. Core Concepts

## 1.1 Agents

Each agent is a neural model:

* A list of weight matrices
* A list of bias vectors
* A list of activations

Example internal agent representation:

```
agent = {
   "weights": [W1, W2, W3],
   "biases": [b1, b2, b3],
   "activations": ["relu", "tanh", "sigmoid"]
}
```

---

# 2. Model Initialization

```
m = MAGE(
    input_dim=features,
    output_dim=labels,
    task="regression" | "binary_classification" | "multiclass",
    initial_hidden=[128, 128],
)
```

### Parameters

| Parameter       | Meaning                                                 |
| --------------- | ------------------------------------------------------- |
| input_dim       | Number of input features                                |
| output_dim      | Output size (1 for binary/regression, N for multiclass) |
| task            | Type of task                                            |
| initial_hidden  | Starting architecture                                   |
| population_size | Number of agents evolved each epoch                     |
| lr              | Local training learning rate                            |

---

# 3. Forward Pass

Each agent performs:

```
for each layer:
    z = xW + b
    a = activation(z)
```

Final output depends on task:

* Regression → identity
* Binary → sigmoid
* Multiclass → softmax

---

# 4. Loss Functions

MAGE uses 3 different losses.

## 4.1 Regression Loss

```
loss = mean((z - y)^2)
grad = 2*(z - y)/N
```

## 4.2 Binary Classification Loss (Safe)

```
p = sigmoid(z)
p = clip(p, 1e-9, 1-1e-9)
loss = -mean(y*log(p) + (1-y)*log(1-p))
grad = (p - y)/N
```

This **prevents NaN** warnings.

## 4.3 Multiclass Loss

```
p = softmax(z)
p = clip(p, 1e-9, 1-1e-9)
loss = -mean(sum(y * log(p)))
grad = (p - y)/N
```

---

# 5. Backpropagation

MAGE implements manual backprop layer-by-layer:

```
dZ = grad_output
for L...0:
    dW = A_prev.T @ dZ
    db = sum(dZ)
    dZ = dZ @ W.T * activation_derivative
```

---

# 6. Genetic Evolution

Each master epoch:

1. Train each agent locally
2. Evaluate fitness on validation set
3. Select top 50%
4. Mutate them
5. Rebuild population

Mutations include:

* Adding/removing hidden layers
* Changing layer size
* Changing activations
* Mutating weights

---

# 7. Training

```
m.fit(X_train, y_train, X_val, y_val,
      master_epochs=10,
      local_epochs=5)
```

Outputs per epoch:

* Validation losses
* Median train losses
* Fitness score

---

# 8. Prediction

### Regression

```
y_pred = m.predict(X)
```

### Binary

```
p = m.predict(X)
y_pred = (p > 0.5).astype(int)
```

### Multiclass

```
probs = m.predict(X)
y_pred = argmax(probs, axis=1)
```

---

# 9. Full Examples

## 9.1 Regression (sklearn make_regression)

```
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=2000, n_features=6, noise=15)
y = y.reshape(-1,1)

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)

m = MAGE(input_dim=6, output_dim=1, task="regression")
m.fit(Xt, yt, Xv, yv, master_epochs=5, local_epochs=5)
```

---

## 9.2 Binary Classification (sklearn make_classification)

```
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=6,
                           n_classes=2, n_informative=4)
y = y.reshape(-1,1)

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)

m = MAGE(input_dim=6, output_dim=1, task="binary_classification")
m.fit(Xt, yt, Xv, yv, master_epochs=5, local_epochs=5)
```

---

## 9.3 Multiclass (sklearn make_classification)

```
X, y = make_classification(n_samples=3000,
                           n_features=6,
                           n_classes=3,
                           n_informative=4)

# one-hot encoding
y_encoded = np.eye(3)[y]

Xt, Xv, yt, yv = train_test_split(X, y_encoded, test_size=0.2)

m = MAGE(input_dim=6, output_dim=3, task="multiclass")
m.fit(Xt, yt, Xv, yv, master_epochs=5, local_epochs=5)
```

---

# 10. Troubleshooting

### **Binary Classification NaN/RuntimeWarning**

Caused by `log(0)`.
Fix is already included using:

```
p = clip(p, 1e-9, 1-1e-9)
```

Make sure this is inside `_loss_and_grad_out`.

---

# 11. Conclusion

This documentation explains:

* How MAGE works internally
* Complete usage examples
* Loss functions
* Evolution process
* Prediction rules
* Fixes for instability

If you want:

* A PDF version
* A cleaned developer manual
* Architecture diagrams

Just say **"export PDF"**.

---

# 12. Why MAGE Is Better (Advantages)

MAGE provides several advantages compared to traditional optimization and neural architecture search (NAS) methods:

## **12.1 It automatically designs architectures**

Traditional ML requires manual tuning of:

* number of layers
* neurons per layer
* activation functions
* initialization schemes

MAGE evolves these automatically, meaning you don't need to guess hyperparameters.

## **12.2 It avoids local minima better than gradient-only optimizers**

Standard training (Adam, SGD) gets stuck in poor regions.
MAGE uses mutation + selection to escape these traps.

## **12.3 It's extremely flexible**

You can evolve:

* Regression networks
* Binary classifiers
* Multiclass classifiers

without changing architecture manually.

## **12.4 More robust on noisy or irregular datasets**

Genetic algorithms naturally handle messy data where gradient-based methods struggle.

## **12.5 Fewer assumptions about loss surface**

Backprop assumes smooth loss surfaces.
Evolutionary methods don't — they can optimize weird, discontinuous spaces.

## **12.6 Allows architecture search + weight training together**

Most NAS systems only search architectures.
MAGE trains weights + evolves structure at the same time.

---

# 13. Where MAGE Can Be Used (Applications)

MAGE is ideal for any domain where:

* The best architecture is unknown
* The data is noisy
* Classical gradient descent performs poorly
* Feature relationships are nonlinear
* You want fast experimentation

## **13.1 Regression Use-Cases**

* Price prediction (housing, stock tendencies, sales)
* Energy consumption forecasting
* Sensor calibration models
* Scientific modeling where functions are unknown

## **13.2 Binary Classification Use-Cases**

* Fraud detection
* Medical diagnosis (yes/no)
* Spam detection
* Churn prediction

## **13.3 Multiclass Classification Use-Cases**

* Image categorization
* Text classification
* Customer segmentation
* Fault detection in machines (multiple failure types)

## **13.4 Research + AutoML**

MAGE is useful to:

* Test new loss functions
* Explore network space
* Prototype new neuroevolution ideas
* Auto-design models without trial-and-error

## **13.5 Small data / messy data environments**

Evolution works even when datasets are tiny or irregular — places where deep learning normally fails.

---

If you want, I can also add:

* A performance comparison section
* A "when NOT to use MAGE" section
* Real world examples with explanations.
