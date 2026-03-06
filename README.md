# SGD-Regressor-from-scratch

The goal of the project is to build an SGD regressor and use it for a recommendation system.

Version 1 of the SGD_v1.ipynb contains the basic model. I have tested this model on the Kaggle Competition playground-series-s5e10. The key issues with this model are that
it does not work on a pandas DataFrame (referring to this code: X_shuffled = X[indices]) since the model tries to select columns by label. NumPy arrays use positional indexing, so X[indices] works perfectly on arrays.

Version 2 of the SGD_v2.ipynb addresses the issues from version 1. X and y get converted into NumPy arrays. To use this model, the features and target label have to be standardized since weights and biases gets exploded to inf/ nans.
