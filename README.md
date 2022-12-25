OPTICS Model build in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- OPTICS
- clustering
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- docker

This is a Clustering Model that uses OPTICS implemented through Sklearn.

OPTICS (Ordering Points To Identify the Clustering Structure) finds core sample of high density and expands clusters from them. It keeps cluster hierarchy for a variable neighborhood radius.

The data preprocessing step includes:

- for numerical variables
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT.

There are no web endpoints provided for this model. Training and prediction is performed by issuing command `train_predict` on the docker container. Also see usage in the `run_local.py` file inside `local_test` directory.
