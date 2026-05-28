
# Preserving Relaxed Dependencies in Data Augmentation: A new RFD-driven Algorithm

We present SyRFD, the first algorithm that preserves Relaxed Functional Dependencies (RFDs) during tabular data generation.


## Installation

Install project requirements

```bash
  pip install -r requirements.txt
```
    
## Project Structure
```
DataAugmentationRFD/
│
├── rfd_augmentation_parametric.py    # SyRFD Algorithm
├── grid_search_classification.py     # Run to perform classification pipeline
├── OllamaAugmentation.ipynb          # Run to perform tuples generation with Ollama
├── OpenRouterAPI.ipynb               # Run to perform tuples generation OpenAI API
├── requirements.txt                   
│
├── imbalanced_datasets/              # Imbalance datasets
│   └── *.csv
├── SMOTE-CDNN/              # Run evaluation with SMOTE-CDNN
│   └── *.py
│   └── smotecdnn_requirements.txt    # SMOTE-CDNN requirements
├── discovered_rfds/                  # Discovered RFDcs
│   └── discovered_rfds_processed/
│       └── RFD*_*.txt
│
├── augmentation_results/             # Tuples resulting after the augmentation 
│   └── *_new_tuples_*.csv
│
├── diff_matrices/                    # Difference matrices
│   └── pw_diff_mx_*.csv
│
├── diff_tuples/                      # IVD matrices
│   └── diff_tuples_*.csv
│
└── classification_results_SYRFD_thr*/           # Classification results obtained by SyRFD and organized by dataset
│
└── evaluation/           # Code and results of the experimental evaluation


```
## Usage/Examples
Before starting the generation process, please set the desired parameters (note that once you have performed data augmentation for a desired threshold, before moving on to the next one, delete all generated files in the folders diff_matrices and diff_tuples)
```python
# SyRFD settings
augmenter = RFDAwareAugmenter(
            imbalance_dataset_path=data_path,
            rfd_file_path=RFD_FILE,
            oversampling=required_train_samples,
            threshold=thr,  # RFDcs similarity threshold
            max_iter=5,  # Maximum attempts per tuple generation
            selected_rfds=None  # Use None for to process all RFDcs, or specify list of rfds to be considered
)
```
To start generating new samples and start the classification with SyRFD, run:
```python
classification_grid_search_syrfd.py
```

To evaluate with other state-of-the-art methods, run:
```python
classification_grid_search_{method}.py
```

Please note that for SMOTE-CDDN you need to install its requiremets, and run:
```python
classification.py
```

To evaluate using the already generated samples (please note you need to change to the desired new_tuples path and the method you want to start the evaluation for), run:
```python
nuovetuple = pd.read_csv(f"classification_results_SYRFD_thr{thr}/new_tuples/{ds}_new_tuples_{thr}.csv") # change the thr accordingly

nuovetuple = pd.read_csv(f"classification_results_{method}/new_tuples/{ds}_new_tuples_{method}.csv") # for the other state-of-the-art methods
```
```python
classification_grid_search_noaug.py

```

## Distributional evaluation

```bash
python check_distribution.py --original ./imbalanced_datasets/iris0.csv --augmented ./augmented_datasets/syrfd8/iris0.csv --alpha 0.05 --output ./distribution_analysis/report_test.csv --plots --plot-dir ./distribution_analysis/plots
python check_distribution.py --original ./imbalanced_datasets/iris0.csv --augmented ./augmented_datasets/syrfd4/iris0.csv --alpha 0.05 --output ./distribution_analysis/report_test.csv --plots --plot-dir ./distribution_analysis/plots
```

## How to cite

Cite the following article: 


@article{cerullo2026aug,
  title={Preserving Relaxed Dependencies in Data Augmentation: A new RFD-driven Algorithm},
  author={Cerullo, M., Cirillo, S., Iuliano, G., Polese, G.},
  journal={TBD},
  year={2026}
}





