import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Define the parameters for the Random Forest model
max_depth = 10
n_estimators = 10

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Mention your experiment below
mlflow.set_experiment('MLOps-Wine-Classification')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy) 

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')       

    # Save the confusion matrix plot
    plt.savefig("confusion_matrix.png")

    # Log the confusion matrix plot as an artifact
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)  # Log the current script as an artifact

    # Tags
    mlflow.set_tags({"Author": 'Asit', "Project": "Wine Classification"})

    # Log the trained model
    mlflow.sklearn.log_model(rf, "RandomForestModel", skops_trusted_types=["sklearn.tree._tree.Tree"])



    print(f"Model trained with accuracy: {accuracy:.4f}")