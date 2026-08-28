# CSC311-Machine-Learning-Project
Classification of Paintings

# Background
Artwork classification is an important application of machine learning, with potential uses in digital archiving, recommendation systems, and art analysis. In this project, we developed models to classify paintings into three categories: *The Persistence of Memory*, *The Starry Night*, and *The Water Lily Pond*. The dataset contained numerical and categorical features describing each painting. We cleaned and transformed these features before training softmax regression and K-nearest neighbours models. To obtain reliable performance estimates, paintings associated with the same unique identifier were kept within the same data split. Cross-validation was then used to select model settings and compare classification performance.

# Project Overview
We explored three model families for classifying survey responses into one of three paintings: K-Nearest Neighbors (KNN), Softmax Regression, and a Multi-Layer Perceptron (MLP).
All models were trained on the same 14-feature set (numeric, Likert, multi-hot categorical,
and TF-IDF text) and tuned via stratified GroupKFold 5-fold cross-validation with macro-F1
as the selection criterion. Our best-performing model is Softmax Regression (LBFGS, C=1),
achieving a projected test accuracy of 89%. Softmax outperformed the MLP (88.2%) and KNN
(85.8%) on the held-out test set, likely because the approximately linear class separation in
our feature space favours a simpler model with fewer parameters to overfit, while KNN’s
distance-based approach struggles in the 172-dimensional feature space.
