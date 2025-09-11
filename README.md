# ARTIFICIAL NEURAL NETWORK (ANN) IMPLEMENTED FROM SCRATCH :

**Author:** SALMA HAMDUN <br>
**Date:** 8/9/2025  <br>
**Version:** 1.0   <br>

______________________________________________________________________________________________________________

## PROJECT OVERVIEW : 
This project predicts customer churn (whether a customer will leave the bank or not) using a neural network built from scratch and a Gradio web app for interactive experimentation , it implements a **3-layer Artificial Neural Network (ANN) from scratch** using **NumPy**, without any high-level deep learning libraries. The network is designed for **binary classification tasks**,demonstrating:

- Forward propagation with ReLU and Sigmoid activations
- Backpropagation with L2 regularization
- Dropout for regularization
- Mini-batch gradient descent with early stopping
- Full training, validation, and testing workflow

______________________________________________________________________________________________________________

## PROJECT'S DEMO :
- It's an interactive Gradio app for training and testing the neural network model on a customer churn prediction task so this DEMO shows a simple run for the neural network based on a customer features. 
- In this DEMO the client can control training parameters : <br> 
                       • **learning rate:** Controls how big each step is during optimization (0.001 → 1.0)
                       • **Epochs:** Number of full passes over the training dataset (1 → 2000)
                       • **hidden units:** Number of neurons in the hidden layer (1 → 64)
- Also , he can control all the **features (inputs) :** <br>
                       •**CreditScore** → Customer’s credit score (as example: 950)
                       •**Geography** → Encoded country (1,2,3)
                       •**Gender** → Encoded gender (0=female , 1=male)
                       •**Age** → Customer’s age
                       •**Tenure** → How many years the customer has been with the bank
                       •**Balance** → Bank account balance
                       •**NumOfProducts** → Number of bank products
                       •**HasCrCard** → (1 : has a credit card , 0 : don't has credit card)
                       •**IsActiveMember** → (1 : active , 0 : not active)
                       •**EstimatedSalary** → Customer’s estimated yearly salary
*(so he can fill them manually and tap submit button & then the magic happens)*

- After submition , the loss curve will appear on the top of prediction , & **prediction will be binary as we hope (0/1)** : 0 means customer will not churn & 1 means he will.
- *The Gradio will automatically open a new browser tab with the app after running the cell.*

- Here's an image which shows how it works.
<img src="Assets/demo.png" alt="demo's run for neural network" width="500"/>

- And here is an option to download an mp4 video to see how to run the gradio of this neural network
[Click here to watch the full video](Assets/demo.mp4)

______________________________________________________________________________________________________________

## 📊 MODEL ACCURACY :
- Training , validation and test accuracy are very close , this means that the model generalizes well without serious overfitting and with good covergence for loss behavior because of early stopping which prevent overfitting by stopping training automatically when the risk of overfit starts.

- It has ~87% testing accuracy , so the model can make strong and good predictions about CHurns. 

- **Training Accuracy:** 86.27%
- **Validation Accuracy:** 87.14%
- **Test Accuracy:** 87.07%

______________________________________________________________________________________________________________

## FEATURES : 
- Fully vectorized **forward and backward propagation**
- **ReLU** activation for hidden layers, **Sigmoid** for output
- **Dropout** in hidden layers for regularization
- **L2 regularization** to prevent overfitting
- **Mini-batch gradient descent** for efficient training
- **Early stopping** based on validation loss
- Metrics calculation: **Accuracy, Confusion Matrix**
- Visualization of **loss curves** and **accuracy bar charts**

______________________________________________________________________________________________________________

## REQUIRED LIBRARIES : 
```python
import pandas as pd   
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import seaborn as sns
```

______________________________________________________________________________________________________________

## NETWORK ARCHITECTURE :
- **Input Layer:** Number of features in the dataset
- **Hidden Layer 1:** ReLU activation, configurable size
- **Hidden Layer 2:** ReLU activation, configurable size
- **Output Layer:** Sigmoid activation, single neuron for binary classification

**Example configuration:**
```python
input_size = X.SHAPE
hidden1_size = 8
hidden2_size = 4
output_size = 1
```

______________________________________________________________________________________________________________

## USAGE INSTRUCTIONS : 

### 1. INITIALISE DATA
```python
# Example: Load dataset and split
X_train, X_val, X_test = ...  # Preprocessed features
y_train, y_val, y_test = ...  # Labels
```

### 2. TRAIN THE MODEL 
```python
trained_params = model(
    X_train, y_train, X_val, y_val,
    input_size, hidden1_size, hidden2_size, output_size,
    epochs=2000, lr=0.2, lambd=0.01, keep_prob=0.9, batch_size=16, patience=100
)
```

### 3. MAKE PREDICTIONS 
```python
pred_train = predict(X_train, trained_params)
pred_val = predict(X_val, trained_params)
pred_test = predict(X_test, trained_params)
```

### 4. EVALUATE PERFORMANCE
```python
print(f"Training Accuracy: {accuracy(y_train, pred_train)*100:.2f}%")
print(f"Validation Accuracy: {accuracy(y_val, pred_val)*100:.2f}%")
print(f"Test Accuracy: {accuracy(y_test, pred_test)*100:.2f}%")
```

### 5. VISUALISE RESULTS
- **Loss Curve**
```python
plt.plot(loss_history, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()
```

- **Accuracy Bar Chart**
```python
train_acc = accuracy(y_train, pred_train) * 100
val_acc = accuracy(y_val, pred_val) * 100
test_acc = accuracy(y_test, pred_test) * 100

plt.bar(["Train", "Validation", "Test"], [train_acc, val_acc, test_acc], color=["green", "orange", "blue"])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")
plt.show()
```

- **Confusion Matrix**
```python
cm = confusion_matrix(y_test.flatten(), pred_test.flatten())
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Exited (0)", "Exited (1)"],
            yticklabels=["Not Exited (0)", "Exited (1)"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Test Set")
plt.show()
```

______________________________________________________________________________________________________________

## PERFORMANCE METRICS :
- Training, validation, and test **accuracy** are reported
- **Confusion matrix** visualizes model errors
- **Loss curve** shows convergence over epochs

______________________________________________________________________________________________________________

## KEY INSIGHTS :
- Dropout and L2 regularization **prevent overfitting**
- Mini-batch gradient descent improves **training efficiency**
- Early stopping ensures **optimal training without excessive epochs**
- ReLU activation avoids vanishing gradient issues in hidden layers

______________________________________________________________________________________________________________

## FUTURE IMPROVEMENTS :
- Implement **Adam or RMSProp** optimizers for faster convergence
- Add **multi-class classification** capability
- Extend to **deeper networks** with more hidden layers
- Add **GPU acceleration** for larger datasets

______________________________________________________________________________________________________________

## 📊EXPLORATORY DATA ANALYSIS (EDA) :
### 1. Target Distribution
- Shows the churn distribution ehich is the 'target' which is 'Exited': 1 or 'Not Exited' : 0
<img src="Assets/churn_distribution.png" alt="Target Distribution" width="500"/>

### 2. Numerical Features
- Shows all the numerical features in the dataset  
<img src="Assets/numerical_features/all_features.png" alt="Numerical features distribution" width="500"/>
<img src="Assets/numerical_features/Age_distribution.png" alt="Age Distribution" width="500"/> 

### 3. Numerical fatures vs Target 
- Shows all numerical features in plotted vs target as scatter 
<img src='Assets/numerical_features/scatter_features.png' alt="Numerical features distribution vs the target" width="500"/>

### 4. HasCard & IsActiveMember features vs Target 
- Shows the two HasCard And IsActiveMember features vs the target "Exited"
<img src='Assets/Categorical_features/hasCard_vs_Exited.png' alt="HasCard features vs Target" width="500"/>
<img src="Assets/Categorical_features/isActiveMember_vs_Exited.png"alt="IsActiveMember vs Target"width="500"/>

### 5. Categorical features vs Target
- Shows all categorical features vs target
<img src="Assets/Categorical_features/curn_by_geography.png" alt="Churn vs Geography" width="500"/>
<img src="Assets/Categorical_features/churn_by_gender.png" alt="Churn vs Gender" width="500"/> 

### 6. Each numerical feature vs Target
- Shows Each Numerical feature in front of Target 
<img src="Assets/numerical_features/Age_vs_churn.png" alt="Age vs Churn" width="500"/> 
<img src="Assets/numerical_features/credit_score_vs_churn.png" alt="Credit Score vs Churn" width="500"/> 

### 7. Correlation Heatmap
- Shows the strength of correlation between all the data columns
<img src="Assets/correlation.png" alt="Correlation Heatmap" width="500"/>

______________________________________________________________________________________________________________

## MODEL VISUALISATION :
### 1. Neural Network Model :
- Shows approximate image of the neural network layers
<img src="Assets/nn.png" alt="Neural network model" width="500"/>

### 2. Cofusion Matrix :
- Shows the number of correct and incorrect predictions for each class & gives insight about where the model performs well and where it makes mistakes.
<img src="Assets/confusion_matrix_test.png" alt="Model's Confusion matrix" width="500"/>

### 3. Loss Curve :
- Shows how the model’s error decreases over epochs & indicates how well it's learning.
<img src="Assets/loss_curve.png" alt="Model's Loss Curve" width="500"/>

### 4. Accuracy :
- Shows the accuracy for each of Training , Validation & Testing Sets in the performance of the model.
<img src="Assets/Model_accuracy.png" alt="Model's Accuracy" width="500"/>

______________________________________________________________________________________________________________

## FILE STRUCTURE :
```
neural_network_project/
├── main.ipynb          # Code + plan + visualisations 
│      
└── Assets/
   ├── Churn_Modelling.csv
   └── Categorical_features
   └── numerical_features

```
______________________________________________________________________________________________________________

## REFERENCES :
- Advanced Learning Algorithms (Stanford) 
- Kaggle
- GitHub
- NumPy and Matplotlib documentation
- Original backpropagation papers

