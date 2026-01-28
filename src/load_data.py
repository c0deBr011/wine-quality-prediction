import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("data/winequality.csv")
print(df.head())
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nRandom 5 rows:")
print(df.sample(5))
print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)
print("\nCheck missing values (True means missing):")
print(df.isnull())

print("\nTotal missing values in each column:")
print(df.isnull().sum())
print("\nQuality value counts:")
print(df['quality'].value_counts())
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df)
plt.title("Distribution of Wine Quality")
plt.xlabel("Wine Quality Score")
plt.ylabel("Count")
plt.show()
# Create binary quality label
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
print("\nQuality vs Quality Label:")
print(df[['quality', 'quality_label']].head(10))
# Separate features and target
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']
print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Initialize the scaler
scaler = StandardScaler()
# Fit scaler on training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform test data
X_test_scaled = scaler.transform(X_test)
print("\nScaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
# Predictions
log_pred = log_model.predict(X_test_scaled)
knn_pred = knn_model.predict(X_test_scaled)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test_scaled)
# Accuracy scores
log_acc = accuracy_score(y_test, log_pred)
knn_acc = accuracy_score(y_test, knn_pred)
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
print("\nModel Accuracy Comparison:")
print(f"Logistic Regression: {log_acc:.2f}")
print(f"KNN: {knn_acc:.2f}")
print(f"Decision Tree: {dt_acc:.2f}")
print(f"Random Forest: {rf_acc:.2f}")
print(f"SVM: {svm_acc:.2f}")
import pandas as pd

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM'],
    'Accuracy': [log_acc, knn_acc, dt_acc, rf_acc, svm_acc]
})

print("\nAccuracy Comparison Table:")
print(results)
# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
# Hyperparameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}
# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
