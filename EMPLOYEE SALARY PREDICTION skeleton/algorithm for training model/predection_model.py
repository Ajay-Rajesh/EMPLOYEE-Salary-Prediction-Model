# ---------- income_train_and_predict.py ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
import joblib

# Load and clean data
df = pd.read_csv("adult 3.csv")
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

# Split features and target
X = df.drop("income", axis=1)
y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# Define columns
num_cols = X.select_dtypes(include=["int64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Remove outliers
from sklearn.ensemble import IsolationForest
isf = IsolationForest(contamination=0.01, random_state=42)
mask = isf.fit_predict(X_processed) != -1
X_filtered = X_processed[mask]
y_filtered = y.iloc[mask].reset_index(drop=True)

if isinstance(X_filtered, csr_matrix):
    X_filtered = X_filtered.toarray()

# Add KMeans cluster as a feature
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_filtered)
X_final = np.hstack([X_filtered, clusters.reshape(-1, 1)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_filtered, test_size=0.2, random_state=42)

# Model setup
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('xgb', xgb)
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy bar plot
plt.figure(figsize=(4, 5))
plt.bar(["Ensemble Accuracy"], [accuracy], color='green')
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.savefig("accuracy_plot.png")
plt.show()

joblib.dump((ensemble, preprocessor, kmeans), "income_ensemble_model.pkl")
