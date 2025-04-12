import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import warnings
warnings.filterwarnings("ignore")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('water_potability')

data = pd.read_csv(
    'https://raw.githubusercontent.com/ty-prct-ai-ds/water_exp/refs/heads/main/data/water_potability.csv')

train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# X_train = train_processed_data.iloc[:, :-1].values
# y_train = train_processed_data.iloc[:, -1].values

X_train = train_processed_data.drop(columns=['Potability'],axis=1)
y_train = train_processed_data['Potability']


rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40],
}

search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                            n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)


# Start the parent MLflow run
with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:

    search.fit(X_train, y_train)

    for i in range(len(search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination {i+1}", nested=True) as child_run:
            params = search.cv_results_['params'][i]
            mean_test_score = search.cv_results_['mean_test_score'][i]

            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", mean_test_score)

    print("Best parameters found: ", search.best_params_)

    mlflow.log_params(search.best_params_)

    best_rf = search.best_estimator_
    best_rf.fit(X_train, y_train)

    pickle.dump(best_rf, open("model.pkl", "wb"))

    # X_test = test_processed_data.iloc[:, :-1].values
    # y_test = test_processed_data.iloc[:, -1].values

    X_test = test_processed_data.drop(columns=['Potability'],axis=1)
    y_test = test_processed_data['Potability']
    
    
    model = pickle.load(open('model.pkl', "rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(search.best_estimator_, "Best Model")

    mlflow.set_tag("author", "Yash Potdar")
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
