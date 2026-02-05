import sklearn.ensemble
import xgboost as xgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss, roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import *
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor


X = np.load("data.npy", allow_pickle=False).astype(np.float64)
y = np.load("labels.npy", allow_pickle=False).astype(np.float64)
X[np.isinf(X)] = np.nan
df = pd.DataFrame(X)
df.interpolate(method='linear', axis=0, inplace=True)
X = df.to_numpy()

test_data = np.load("sub_data.npy", allow_pickle=False).astype(np.float64)
test_data[np.isinf(test_data)] = np.nan
df = pd.DataFrame(test_data)
df.interpolate(method='linear', axis=0, inplace=True)
test_data = df.to_numpy()

test_data2 = np.load("sub_dataw.npy", allow_pickle=False).astype(np.float64)
test_data2[np.isinf(test_data2)] = np.nan
df = pd.DataFrame(test_data2)
df.interpolate(method='linear', axis=0, inplace=True)
test_data2 = df.to_numpy()

test_labels = np.load("sub_labels.npy", allow_pickle=False).astype(np.float64)


X_clean = X
y_clean = y

test_data = test_data
test_labels = test_labels

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.001, random_state=42)

p = [1] * 131407
sub_file = pd.read_csv("/home/lm/Downloads/proje/marchmania/march-machine-learning-mania-20252/SampleSubmissionStage2.csv")


def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred2 = pipeline.predict(test_data)
    y_pred3  = pipeline.predict(test_data2)

    y_prob2 = pipeline.predict_proba(test_data)[:, 1] if hasattr(pipeline, "predict_proba") else None
    y_prob3 = pipeline.predict_proba(test_data2)[:, 1] if hasattr(pipeline, "predict_proba") else None
    preds = np.concatenate((y_prob2, y_prob3))
    sub_file["Pred"] = preds
    sub_file.to_csv("real_submission.csv", index=False)

    print("Predicted probabilities for test_data:")
    print(y_prob2)
    print("Shape of predicted probabilities for test_data2:")
    print(y_prob3.shape)

    results = {
        'model_name': model.__class__.__name__,
        'accuracy': accuracy_score(test_labels, y_pred2),
        'precision': precision_score(test_labels, y_pred2),
        'recall': recall_score(test_labels, y_pred2),
        'f1': f1_score(test_labels, y_pred2),
        "brier_score": round(brier_score_loss(test_labels, y_prob2), 3)
    }

    if y_prob2 is not None:
        results['roc_auc'] = roc_auc_score(test_labels, y_prob2)

    return results, pipeline



models = [
    HistGradientBoostingClassifier(random_state=42, max_iter=1000, max_depth=1),
]


results = []
trained_models = {}

for model in models:
    print(f"Training {model.__class__.__name__}...")
    model_results, trained_pipeline = train_evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append(model_results)
    trained_models[model.__class__.__name__] = trained_pipeline
    print(
        f"{model.__class__.__name__} completed. "
        f"Brier Score: {model_results['brier_score']:.10f} "
        f"F1 Score: {model_results['f1']:.10f}"
    )


results_df = pd.DataFrame(results)
results_df = results_df.set_index('model_name')
print("\nModel comparison results:")
print(results_df)


best_model_name = "XGBClassifier"
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name} (F1 Score: {results_df.loc[best_model_name, 'f1']:.10f})")


y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Prediction')
plt.show()


import joblib

joblib.dump(best_model, f'best_model_{best_model_name}.pkl')


print("\nStarting hyperparameter optimization for the best model...")

if best_model_name == "LogisticRegression":
    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'lbfgs'],
        'model__penalty': ['l1', 'l2']
    }
elif best_model_name == "RandomForestClassifier":
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == "GradientBoostingClassifier":
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5]
    }
elif best_model_name == "SVC":
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__gamma': ['scale', 'auto', 0.1, 0.01]
    }
elif best_model_name == "XGBClassifier":
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6],
        'model__subsample': [0.8, 0.9, 1.0]
    }


grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)


print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


optimized_model = grid_search.best_estimator_
y_pred_opt = optimized_model.predict(test_data)
opt_accuracy = accuracy_score(test_labels, y_pred_opt)
opt_precision = precision_score(test_labels, y_pred_opt)
opt_recall = recall_score(test_labels, y_pred_opt)
opt_f1 = f1_score(test_labels, y_pred_opt)
brier = brier_score_loss(test_labels, y_pred_opt)

print("\nOptimized model test set performance:")
print(f"Accuracy: {opt_accuracy:.4f}")
print(f"Precision: {opt_precision:.4f}")
print(f"Recall: {opt_recall:.4f}")
print(f"F1 Score: {opt_f1:.4f}")
print(f"Brier Score: {brier:.4f}")


joblib.dump(optimized_model, f'optimized_model_{best_model_name}.pkl')


if hasattr(optimized_model, 'feature_importances_') or (
        hasattr(optimized_model, 'named_steps') and hasattr(optimized_model.named_steps['model'],
                                                            'feature_importances_')):
    if hasattr(optimized_model, 'feature_importances_'):
        importances = optimized_model.feature_importances_
    else:
        importances = optimized_model.named_steps['model'].feature_importances_

    if X_train.shape[1] <= 20:
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), indices)
        plt.tight_layout()
        plt.show()

        print("\nMost important features:")
        for i, idx in enumerate(indices[:10]): 
            print(f"{i + 1}. Feature {idx}: {importances[idx]:.4f}")
