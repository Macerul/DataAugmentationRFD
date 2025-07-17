import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
from imblearn.metrics import geometric_mean_score



"""
The G-mean, or geometric mean, is a performance metric in machine learning used for evaluating binary
 classification models, particularly in imbalanced datasets. 
 It provides a balanced view by considering both true positive rate (sensitivity) and 
 true negative rate (specificity). 
 The G-mean is calculated as the square root of the product of sensitivity and specificity. 

Imbalanced Data:
It's particularly useful for evaluating classifiers on imbalanced datasets, 
where one class has significantly more samples than the other.
"""


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file. Assumes the target column is named 'class'.
    """
    return pd.read_csv(path)


def plot_confusion_matrix(cm: pd.DataFrame, classes: list, title: str, filename: str):
    """
    Plot and save the confusion matrix.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Parameters
    data_path = 'imbalanced_datasets/vehicle0.csv'
    basename = os.path.basename(data_path)
    dname = os.path.splitext(basename)[0]
    target_col = 'class'
    test_size = 0.3
    random_state = 42

    # Load data
    df = load_data(data_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state)
    }

    metrics = []

    cm_dir = 'classification_results'
    os.makedirs(cm_dir, exist_ok=True)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        gmean = geometric_mean_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm,
            classes=[f'Class {c}' for c in sorted(df[target_col].unique())],
            title=f'{name} Confusion Matrix',
            filename=os.path.join(cm_dir, f"{name.replace(' ', '_')}_cm_{dname}.png")
        )

        metrics.append({
            'Model': name,
            'AUC': auc,
            'F1-score': f1,
            'G-mean': gmean,
            'Accuracy': acc
        })

    results_df = pd.DataFrame(metrics)
    results_csv = f'{dname}_classification_results.csv'
    results_df.to_csv(os.path.join(cm_dir,results_csv), index=False)
    print(f"Results saved to {results_csv}")
    print(results_df)


if __name__ == '__main__':
    main()
