# Machine Learning Practicals

Each experiment has its own folder containing `main.py` and its CSV dataset:

- `experiment_1_housing/` - Boston Housing linear regression
- `experiment_2_titanic/` - Titanic logistic regression
- `experiment_3_adult_decision_tree/` - Adult Census decision tree
- `experiment_4_adult_random_forest/` - Adult Census random forest
- `experiment_5_adult_adaboost/` - Adult Census AdaBoost
- `experiment_6_wholesale_clustering/` - Wholesale customer hierarchical clustering

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Run

From the workspace root, run an experiment with its script path, for example:

```bash
python experiment_1_housing/main.py
```

Replace the folder name to run a different experiment. Plots open in a window when supported by the active Python environment.

Each script prompts for a custom example after training and evaluation. For categorical Adult Census and Titanic values, choose from the options printed by the script. The clustering experiment assigns a new customer to the nearest cluster centroid as an approximation.

The CSV names inside each folder match the filenames loaded by the scripts. The Adult Census CSV is included separately in experiments 3, 4, and 5 so each folder is self-contained.