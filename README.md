# CSC311-Machine-Learning-Project
Classification of Paintings
We explored three model families for classifying survey responses into one of three paintings: K-Nearest Neighbors (KNN), Softmax Regression, and a Multi-Layer Perceptron (MLP).
All models were trained on the same 14-feature set (numeric, Likert, multi-hot categorical,
and TF-IDF text) and tuned via stratified GroupKFold 5-fold cross-validation with macro-F1
as the selection criterion. Our best-performing model is Softmax Regression (LBFGS, C=1),
achieving a projected test accuracy of 89%. Softmax outperformed the MLP (88.2%) and KNN
(85.8%) on the held-out test set, likely because the approximately linear class separation in
our feature space favours a simpler model with fewer parameters to overfit, while KNN’s
distance-based approach struggles in the 172-dimensional feature space.
