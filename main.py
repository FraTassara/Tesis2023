import os
import subprocess

def runExperiment(dataset_name):
    """Executes random search and final training for a given dataset."""
    print(f"\n*** Running experiment for {dataset_name} ***")
    
    # Run hyperparameter search
    print("Running hyperparameter search...")
    subprocess.run(["python", f"experiments/{dataset_name}/random_search.py"])
    
    # Run final training
    print("Running final training with optimal parameters...")
    subprocess.run(["python", f"experiments/{dataset_name}/train.py"])

if __name__ == "__main__":
    # List of datasets
    datasets = ["dataset_1", "dataset_2"]
    
    # Execute experiments for each dataset
    for dataset in datasets:
        runExperiment(dataset)