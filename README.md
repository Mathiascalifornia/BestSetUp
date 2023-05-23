# BestSetUp

This class aims to save time by giving you the best options for a classification or regression problem,
in terms of model, scaler / transformer, preprocessing of categorical data, and if managing outliers is useful or not.
All the results are found using a KFold or StratifiedKFold cross validation (you have to specify how many).
Only works with sklearn compatible models (xgboost and catboost included).
You can also retrieve the preprocessed dataset via the 'get_X_and_y' method.
