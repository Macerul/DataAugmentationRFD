import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from numpy.ma.core import count
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from rfd_augmentation_parametric import RFDAwareAugmenter

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
    data_path = 'imbalanced_datasets/wisconsin.csv'
    RFD_FILE = 'discovered_rfds/discovered_rfds_processed/RFD12_E0.0_wisconsin_min.txt'
    basename = os.path.basename(data_path)
    dname = os.path.splitext(basename)[0]
    target_col = 'class'
    test_size = 0.3
    random_state = 42


    df = load_data(data_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


    print('Training data shape:', X_train.shape)
    print('Training positive samples shape:', list(y_train).count(1))
    print('Training negative samples shape:', list(y_train).count(0))

    required_train_samples = list(y_train).count(0) - list(y_train).count(1)
    print('Required training positive samples:', required_train_samples)


    print('Test data shape:', X_test.shape)
    print('Test positive samples shape:', y_test.value_counts())


    # CONFIGURE AUGMENTER PARAMETERS
    augmenter = RFDAwareAugmenter(
        imbalance_dataset_path=data_path,
        rfd_file_path=RFD_FILE,
        oversampling=required_train_samples,
        threshold=12,  # RFD similarity threshold
        max_iter=5,  # Maximum attempts per tuple generation
        selected_rfds=None  # Use None for all RFDs, or specify list of rfds to be considered
    )



    # Run augmentation with
    X_train_new_pos = augmenter.augment_dataset()
    print('Shape df delle nuove tuple aggiunte:', X_train_new_pos.shape)
    y_train_new_pos = X_train_new_pos[target_col]
    print(y_train_new_pos.head())
    print('Shape df delle nuove tuple aggiunte (target column):', y_train_new_pos.shape)
    X_train_new_pos.drop(target_col, axis=1, inplace=True)
    print(X_train_new_pos.head())
    print('Shape df delle nuove tuple aggiunte:', X_train_new_pos.shape)

    # TRAIN DATA RISULTANTI DOPO AVER APPLICATO LA STRATEGIA DI DATA AUGMENTATION
    X_train_augmented = pd.concat([X_train, X_train_new_pos], ignore_index=True)
    y_train_augmented = pd.concat([y_train, y_train_new_pos], ignore_index=True)

    # Fit SMOTE on the training data
    smote = SMOTE(random_state=42)
    #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'XGBoost': xgb.XGBClassifier(random_state=random_state),
    }

    metrics = []

    cm_dir = 'classification_results'
    os.makedirs(cm_dir, exist_ok=True)

    for name, model in models.items():
        # Train
        model.fit(X_train_augmented, y_train_augmented)
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        gmean = geometric_mean_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm,
            classes=[f'Class {c}' for c in sorted(df[target_col].unique())],
            title=f'{name} Confusion Matrix',
            filename=os.path.join(cm_dir, f"{name.replace(' ', '_')}_cm_{dname}_aug.png")
        )

        metrics.append({
            'Model': name,
            'AUC': "%.2f" % auc,
            'F1-score': "%.2f" % f1,
            'Precision': "%.2f" % prec,
            'Recall': "%.2f" % rec,
            'G-mean': "%.2f" % gmean,
            'Accuracy': "%.2f" % acc
        })

    results_df = pd.DataFrame(metrics)
    results_csv = f'{dname}_classification_results_aug.csv'
    results_df.to_csv(os.path.join(cm_dir,results_csv), index=False)
    print(f"Results saved to {results_csv}")
    print(results_df)


if __name__ == '__main__':
    main()