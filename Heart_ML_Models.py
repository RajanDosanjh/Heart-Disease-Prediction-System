# ===========================================
# Heart Disease Prediction - Full Pipeline
# ===========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# ===========================================
#  Load and Inspect Dataset
# ===========================================
file_path = "heart.csv"
data = pd.read_csv(file_path)

print("Initial Shape:", data.shape)

# Drop duplicates (important for your dataset)
data = data.drop_duplicates()
print("After Removing Duplicates:", data.shape)

print("\nMissing Values:\n", data.isnull().sum())

# ===========================================
#  Outlier Capping
# ===========================================
def cap_outliers(column):
    q1, q3 = column.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return column.clip(lower=lower_bound, upper=upper_bound)

numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    if col != "target":
        data[col] = cap_outliers(data[col])

# ===========================================
# Encode Categorical Variables
# ===========================================
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# ===========================================
# Feature Engineering
# ===========================================
data['age_group'] = pd.cut(data['age'], bins=[0, 40, 60, 80], labels=['Young', 'Middle-aged', 'Senior'])
data['heart_stress'] = data['trestbps'] / data['thalach']
data['risk_score'] = data['chol'] * data['trestbps']

data = pd.get_dummies(data, columns=['age_group'], drop_first=True)

# ===========================================
# Split features/Target
# ===========================================
X = data.drop('target', axis=1)
y = data['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================================
# Feature Importance
# ===========================================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance from Random Forest")
plt.gca().invert_yaxis()
plt.show()

# ===========================================
# SelectKBest Statistical Selection
# ===========================================
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("\nSelected Features (Statistical):", list(selected_features))

# Combine RandomForest + SelectKBest
important_features = feature_importance.loc[feature_importance['Importance'] > 0.01, 'Feature'].tolist()
final_features = list(set(list(selected_features) + important_features))

X_final = X[final_features]
print("\nFinal Features for Modeling:", final_features)

# ===========================================
# Model Training & Evaluation
# ===========================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n===== {model_name} =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    return model

#KNN (with Pipeline + GridSearch) 
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
param_grid = {'knn__n_neighbors': list(range(1, 31))}
grid_knn = GridSearchCV(pipeline_knn, param_grid, cv=5, scoring='f1')
grid_knn.fit(X_train, y_train)

best_k = grid_knn.best_params_['knn__n_neighbors']
print("\nBest k for KNN:", best_k)
evaluate_model(grid_knn.best_estimator_, X_train, X_test, y_train, y_test, "KNN")

#Decision Tree
dt_params = {
    "max_depth": [2, 5, 10, 15],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 5, 10]
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring="f1")
grid_dt.fit(X_train, y_train)
print("\nOptimal DT Params:", grid_dt.best_params_)
evaluate_model(grid_dt.best_estimator_, X_train, X_test, y_train, y_test, "Decision Tree")

# Logistic Regression
logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])
evaluate_model(logreg, X_train, X_test, y_train, y_test, "Logistic Regression")

# Gaussian Naive Bayes
gnb = GaussianNB()
evaluate_model(gnb, X_train, X_test, y_train, y_test, "Naive Bayes")

