import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("E:\Git\Fraud-transaction-detection\Data\Fraud.csv")
df = df.drop(['step', 'nameDest', 'nameOrig', 'isFlaggedFraud'], axis=1)

df['orig_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']

categorical_cols = ['type']
numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                  'oldbalanceDest', 'newbalanceDest',
                  'orig_balance_change', 'dest_balance_change']

X = df[categorical_cols + numerical_cols]
y = df['isFraud']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(probability=True)
}

best_model = None
best_score = 0

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print(f"\n🔍 {name} Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = pipeline
        best_name = name

joblib.dump(best_model, f"E:\Git\Fraud-transaction-detection\Model\best_pipeline_{best_name}.pkl")
print(f"\n✅ Best model ({best_name}) saved as 'best_pipeline_{best_name}.pkl'")