import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
dagshub.init(repo_owner='yashpotdar-py',
             repo_name='mlflow_exp_dagshub', mlflow=True)

mlflow.set_experiment("water_exp")
mlflow.set_tracking_uri(
    "https://dagshub.com/yashpotdar-py/mlflow_exp_dagshub.mlflow")
data = pd.read_csv(
    'https://raw.githubusercontent.com/ty-prct-ai-ds/water_exp/refs/heads/main/data/water_potability.csv')

train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

X_train = train_processed_data.iloc[:, 0:-1].values
y_train = train_processed_data.iloc[:, -1].values

n_estimators = 1000

with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    # save
    pickle.dump(clf, open("model.pkl", "wb"))

    X_test = test_processed_data.iloc[:, 0:-1].values
    y_test = test_processed_data.iloc[:, -1].values

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    mlflow.log_param("n_estimators", n_estimators)

    mlflow.sklearn.log_model(clf, "GradientBoostingClassifier")
    mlflow.log_artifact(__file__)

    mlflow.set_tag("author", "Yash Potdar")
    mlflow.set_tag("model", "GradientBoostingClassifier")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
