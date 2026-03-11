# SGD-Regressor-from-scratch

The goal of the project is to build a recommendation system from scratch, starting with an SGDRegressor. To build an intuition about how this model works, we need to understand the math behind it.

Before we dive into the project, let me explain what gradient descent is. Gradient descent is an optimization algorithm that minimizes a cost function by iteratively moving in the direction of the steepest descent. Basically, you start with a random point on the mountain, find the slope, and try to minimize the cost by following in that direction.
The size of each step is determined by the learning rate. The stochastic nature of this algorithm refers to the randomness.

Version 1 of the SGD_v1.ipynb contains the basic model. I have tested this model on the Kaggle Competition playground-series-s5e10. The key issues with this model are that
it does not work on a pandas DataFrame (referring to this code: X_shuffled = X[indices]) since the model tries to select columns by label. NumPy arrays use positional indexing, so X[indices] works perfectly on arrays.

Version 2 of the SGD_v2.ipynb addresses the issues from version 1. X and y get converted into NumPy arrays. To use this model, the features and target label need to be standardized, since the weights and biases can explode to inf/nan. Scaling the features helps gradient descent converge quickly and reduces computational time. We have created mini-batch SGD, as it tends to have a smoother descent than single batch SGD.
