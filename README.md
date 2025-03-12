# Decision Tree Classifier for Income Prediction

## Overview
This project implements a **Decision Tree Classifier** on adult data set (https://archive.ics.uci.edu/ml/datasets/Adult) from the
UCI Machine Learning Repository to predict whether an individual's income is **≤ 50K or > 50K** based on attributes like age, working hours, and education level. The model is trained on given data, and the training error rate is computed.

## Requirements
Install dependencies using:
```bash
pip install pandas scikit-learn
```

## Usage
### 1. Load Data and Pre-process

### 2. Train and Predict
```python
from decision_tree_classifier import dt_predict

predictions = dt_predict(X_train, y_train)
print(predictions)
```

### 3. Expected Output
- A **pandas Series** of predicted labels (`<=50K` or `>50K`).
- **Training error rate** displayed.

Example:
```
Training Error Rate: 0.0000
0    <=50K
1     >50K
2     >50K
3    <=50K
```

## Function Overview
### `dt_predict(X_train, y_train)`
1. Trains a **Decision Tree Classifier**.
2. Predicts labels for `X_train`.
3. Computes **training error rate**.
4. Returns a **pandas Series** with predictions.

## Customization by Changing Classifier Parameters
Modify `DecisionTreeClassifier` parameters for better performance:
```python
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
```
- `max_depth=5`: Prevents overfitting.
- `min_samples_split=10`: Requires at least 10 samples for a split.

## License
Open-source and free to use.
