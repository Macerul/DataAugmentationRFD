import os
import re

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, \
    balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
import warnings


warnings.filterwarnings('ignore')
# from rfd_augmentation_parametric import RFDAwareAugmenter
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
    return pd.read_csv(path, sep=',')


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


def get_models_and_params():
    """
    Define models and their corresponding hyperparameter grids for GridSearch.
    """
    models_params = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'algorithm': ['SAMME']
            }
        },
        # NUOVI MODELLI AGGIUNTI
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },

        'Neural Network (MLP)': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.15, 0.5, 0.7, 0.9]
            }
        }
    }

    return models_params


def perform_grid_search(model, param_grid, X_train, y_train, cv=3, scoring='f1'):
    """
    Perform grid search with cross-validation to find the best hyperparameters.
    """
    print(f"Performing grid search for {model.__class__.__name__}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_:.2f}")

    return grid_search.best_estimator_


def main():
    # Parameters
    metodo = "deepseek"
    thr = 2
    '''"kddcup-guess_passwd_vs_satan","Migraine_onevsrest_0","Migraine_onevsrest_1",
                "Migraine_onevsrest_2","Migraine_onevsrest_3","Migraine_onevsrest_4",
                "Migraine_onevsrest_5","new-thyroid1","newthyroid2","Obesity_onevsrest_0",
                "Obesity_onevsrest_1",'''

    aug_lama_dir = f"augmented_datasets_LLM/{metodo}"
    datasets = os.listdir(aug_lama_dir)
    # grandi: "abalone19",
    #datasets = ["ecoli-0_vs_1","Migraine_onevsrest_0"]
    for ds in datasets:
        print(ds)
        data_path = f'imbalanced_datasets/{ds}'
        #RFD_FILE = f'discovered_rfds/discovered_rfds_processed/RFD{thr}_E0.0_{ds}_min.txt'
        #m = re.search(r'RFD(\d+)_', RFD_FILE)
        #if m:
            #thr = int(m.group(1))
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

        counts = y_train.value_counts()
        max_count = counts.max()
        min_count = counts.min()

        required_train_samples = max_count - min_count
        print('Required training positive samples:', required_train_samples)
        # print('Test data shape:', X_test.shape)
        # print('Test positive samples shape:', y_test.value_counts())

        nuovetuple = pd.read_csv(f"augmented_datasets_LLM/{metodo}/{dname}.csv", sep=',')

        print(dname)
        # Run augmentation with
        if required_train_samples > len(nuovetuple):
            X_train_new_pos = nuovetuple.sample(len(nuovetuple))
        else:
            X_train_new_pos = nuovetuple.sample(required_train_samples)
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

        # Get models and hyperparameters
        models_params = get_models_and_params()

        metrics = []
        best_models = {}
        cm_dir = f'classification_results_{metodo}'
        os.makedirs(cm_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("STARTING GRID SEARCH AND MODEL EVALUATION")
        print("=" * 60)

        for name, model_config in models_params.items():
            print(f"\n{'=' * 20} {name} {'=' * 20}")

            # Perform grid search
            best_model = perform_grid_search(
                model=model_config['model'],
                param_grid=model_config['params'],
                X_train=X_train_augmented,
                y_train=y_train_augmented,
                cv=3,
                scoring='f1'
            )

            best_models[name] = best_model

            # Train with best parameters
            print(f"Training {name} with best parameters...")
            best_model.fit(X_train_augmented, y_train_augmented)

            # Predict
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            # Compute metrics
            auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            gmean = geometric_mean_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            print(f"Results - AUC: {auc:.2f}")

            # Save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(
                cm,
                classes=[f'Class {c}' for c in sorted(df[target_col].unique())],
                title=f'{name} Confusion Matrix (Grid Search)',
                filename=os.path.join(cm_dir, f"{name.replace(' ', '_')}_cm_{dname}_aug_gridsearch_{metodo}.pdf")
            )

            metrics.append({
                'Model': name,
                'AUC': "%.2f" % auc,
                'F1-score': "%.2f" % f1,
                'Precision': "%.2f" % prec,
                'Recall': "%.2f" % rec,
                'G-mean': "%.2f" % gmean,
                'Balanced-Accuracy': "%.2f" % bal_acc,
                'Accuracy': "%.2f" % acc
            })

        # Save results
        results_df = pd.DataFrame(metrics)
        results_csv = f'{dname}_classification_results_aug_gridsearch_{metodo}.csv'
        results_df.to_csv(os.path.join(cm_dir, results_csv), index=False)

        print(f"\n{'=' * 60}")
        print("FINAL RESULTS")
        print(f"{'=' * 60}")
        print(f"Results saved to {results_csv}")
        print(results_df.to_string(index=False))

        # Save best models parameters
        best_params_df = []
        for name, model in best_models.items():
            params = model.get_params()
            best_params_df.append({
                'Model': name,
                'Best_Parameters': str(params)
            })

        best_params_results = pd.DataFrame(best_params_df)
        params_csv = f'{dname}_best_parameters_gridsearch_{thr}.csv'
        best_params_results.to_csv(os.path.join(cm_dir, params_csv), index=False)
        print(f"\nBest parameters saved to {params_csv}")

        # Find best performing model
        results_df_numeric = results_df.copy()
        for col in ['AUC', 'F1-score', 'Precision', 'Recall', 'G-mean', 'Balanced-Accuracy', 'Accuracy']:
            results_df_numeric[col] = results_df_numeric[col].astype(float)

        best_f1_idx = results_df_numeric['F1-score'].idxmax()
        best_auc_idx = results_df_numeric['AUC'].idxmax()

        print(f"\n{'=' * 60}")
        print("BEST PERFORMING MODELS")
        print(f"{'=' * 60}")
        print(f"Best F1-score: {results_df.iloc[best_f1_idx]['Model']} ({results_df.iloc[best_f1_idx]['F1-score']})")
        print(f"Best AUC: {results_df.iloc[best_auc_idx]['Model']} ({results_df.iloc[best_auc_idx]['AUC']})")


if __name__ == '__main__':
    main()