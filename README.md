# Readme for "AsymMirai: Interpretable Breast Cancer Risk Prediction from Mammograms"

# Introduction
This repository was used to develop AsymMirai.
The codebase itself is a fork of the MIT-licensed Mirai codebase (https://github.com/yala/Mirai).
Because AsymMirai also relies on the Mirai backbone, most of the Mirai code is retained in this repository.
See [README_mirai.md](./README_mirai.md) for details on the parts of the codebase original to Mirai.

AsymMirai was primarily trained and evaluated on the EMory BrEast imaging Dataset (EMBED); as such, it assumes the information described in [the EMBED documentation](https://github.com/Emory-HITI/EMBED_Open_Data) is available.
This dataset is available upon request; instructions to gain access are available in the EMBED documentation.

# Data Format
This implementation assumes there exists a CSV file of the form shown in `example_data_format.csv` with the path to each image and metadata for a series of exams.

# Training AsymMirai
The jupyter notebook `asymmetry_model/run_train.ipynb` provides an example of the call to train AsymMirai. In order to train AsymMirai, you will need to download the publicly trained weights from Mirai's backbone and place them in a directory named `snapshots` in the root directory for this repository. These weights are available [here](https://duke.box.com/s/g21ak9kfneudotokp9nmlp07r1bsdki7).

# Evaluating AsymMirai
The jupyter notebook `asymmetry_model/run_eval.ipynb` provides an example of the code used to evaluate a trained AsymMirai. This code runs AsymMirai over each exam in the indicated CSV file, and records the risk score and prediction window location for each sample. Our trained AsymMirai weights are available [here](https://duke.box.com/s/9uu9sarz6zizjkqj41iavgxz6zxwxz7c).

# Analyzing AsymMirai
The jupyter notebook `additional_asymmirai_experiments.ipynb` provides the code used to analyze AsymMirai, producing ROC curves.
Note: The values for "percent window shift" computed in this notebook previously had a scaling error. Each window shift percent reported in the paper should be scaled down by a factor of 5/7, meaning the AUC reported at 50% window shift actually corresponds to 36% window shift. No other values -- e.g., the AUC when a given number of patients are included -- are effected.

# Deploying AsymMirai
AsymMirai is open to access and use. If you use this model, please attribute credit to this group. Additionally, we like to hear about this model's use -- if you have successfully used AsymMirai, feel free to reach out with updates on your progress. We are especially interested in quantitative metrics around AsymMirai's uptake.
