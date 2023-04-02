
import pandas as pd , numpy as np 
from sklearn.preprocessing import MinMaxScaler
from pyod.models.iforest import IForest
from sklearn.model_selection import cross_val_score


class BestSetUp:

    """
      
    This class aims to save time by giving you the best options for a datascience problem,
    in terms of model, scaler / transformer, preprocessing of categorical data, and if managing outliers is useful or not.
    All the results are found using cross validation.
    Attributes:
    X : A proper dataframe of explanatory variables (with no missing values , and the unuseful features removed).
    y : The target variable in a dataframe with one column.
    models : A list of models , with every models able to functioning with a sklearn fashion
    scalers : A list of sklearn scalers
    n_cv : Number of cross validation to perform at each step. More cv increase the computing time.
    Methods:
    chose_model : Chose the best model for the dataset , using MinMaxScaler and One-hot-encoding as preprocessing steps
    chose_best_scaler : Chose the best scaler , using only the numerical columns
    chose_between_target_encoding_and_ohe : Chose between one-hot-encoding and target encoding (only for regression problems , otherwise ohe will be used)
    with_or_without_outliers : Using an IForest algorithm , compare the scores between the dataframe without the 5 per cent most extreme values , and the original
    get_best_setup : Returns a string with the best set up found for your problem
    get_X_and_y : Returns X and y preproccessed with the new set up
    
    """

    def __init__(self , X : pd.DataFrame , y : pd.DataFrame , models : list , scalers : list , n_cv : int):
        self.X = X 
        self.y = y
        self.y.columns = ['target']
        self.models = models 
        self.scalers = scalers 
        self.n_cv = n_cv

        if self.y.values.dtype == np.number and float(self.y.nunique()) >= 5: # For regression problems
            self.scoring = 'neg_mean_squared_error'
        else:
             self.scoring = 'accuracy' # For classification


    def chose_model(self):
        
        # Preprocess categorical data for testing purposes
        cat_cols = self.X.select_dtypes(include=object)    
        X_ = pd.concat([pd.get_dummies(cat_cols) , self.X.select_dtypes(include=np.number)] , axis=1)  
        
        # Scale the numerical features using MinMaxScaler
        for col in X_.select_dtypes(include = np.number):
            if X_[col].nunique() >= 5:
                X_[col] = MinMaxScaler().fit_transform(X_[col].values.reshape(-1,1))


        best_model = self.models[0]
        best_score = np.mean(cross_val_score(best_model , X_, self.y, cv=self.n_cv, scoring=self.scoring))
        
        
        for model in self.models:
                
                # Evaluate models
                scores = cross_val_score(model, X_, self.y, cv=self.n_cv, scoring=self.scoring)
                
                # Mean score
                mean_score = np.mean(scores)
                
                # Check if the current best score 
                if mean_score > best_score:
                    best_model = model
                    best_score = mean_score
            
        self.best_model = best_model




    def chose_best_scaler(self):
  
        # Work only with the numerical features
        to_drop = []
        for col in self.X.select_dtypes(include=np.number):
            if self.X[col].nunique() <= 2: # Not binary
                to_drop.append(col)
        
        # Drop the ordinal / binary / columns with not enought diversity
        X_num = self.X.drop(to_drop , axis=1)
        X_num = X_num.select_dtypes(include=np.number)

        best_scaler = self.scalers[0] 
        best_score = np.mean(cross_val_score(estimator=self.best_model , X=X_num , y=self.y , cv=self.n_cv , scoring=self.scoring))


        for scaler in self.scalers:
            X_num_copy = X_num.copy()
            X_num_copy[X_num_copy.columns] = scaler.fit_transform(X_num_copy[X_num_copy.columns])
            scores = cross_val_score(estimator=self.best_model , X=X_num_copy , y=self.y , cv=self.n_cv , scoring=self.scoring)
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_scaler = scaler
                best_score = mean_score

        self.best_scaler = best_scaler
        self.X[X_num.columns] = best_scaler.fit_transform(self.X[X_num.columns])
    


    def chose_between_target_encoding_and_ohe(self):
        # Transform X using the best scaler
        for col in self.X.select_dtypes(include=np.number):
            if self.X[col].nunique() <= 5:
                self.X[col] = self.best_scaler.fit_transform(self.X[col].values.reshape(-1,1))

        # X one hot encode
        dummies = pd.get_dummies(self.X.select_dtypes(include=object))
        X_ohe = pd.concat([self.X , dummies] , axis='columns')
        X_ohe = X_ohe.drop(list(X_ohe.select_dtypes(include=object).columns) , axis=1 , errors='ignore')
        X_ohe['target'] = self.y.values


        if self.scoring == 'accuracy':
            self.best_cat_prepro = 'One-Hot-Encoding' # For binary or label encoding it should be done with domain knowledge , before using this clas
            self.X = X_ohe
        
        else: # Because target encoding works only for regression problems

            X_te = self.X.copy() # The target encoding dataframe
            X_te['target'] = self.y.values

            for col in X_te.select_dtypes(include=object):
                means_values = dict(X_te.groupby(col)['target'].mean())
                X_te[col] = [means_values.get(val) for val in list(X_te[col])]

            
            target_encoding_score = np.mean(cross_val_score(estimator=self.best_model , X=X_te.drop('target' , axis=1) , y=self.y , cv=self.n_cv , n_jobs=-1))
            ohe_score = np.mean(cross_val_score(estimator=self.best_model , X=X_ohe.drop('target' , axis=1) , y=self.y , cv=self.n_cv , n_jobs=-1))


            if target_encoding_score > ohe_score:
                self.X = X_te
                self.best_cat_prepro = 'Target encoding'
                
            else:
                self.X = X_ohe
                self.best_cat_prepro = 'One-Hot_Encoding'

        


    def with_or_without_outliers(self):
            iforest = IForest(contamination=0.05) # Drop the most extreme 5 per cent of the dataframe
            iforest.fit(self.X)
            is_outliers = list(iforest.predict(self.X))


            self.X['outlier'] = is_outliers
            self.y['outlier'] = is_outliers
            
            X_without_outliers = self.X[self.X['outlier'] != 1]
            y_without_outliers = self.y[self.y['outlier'] != 1]

            X_without_outliers = X_without_outliers.drop('outlier' , axis=1)
            y_without_outliers = y_without_outliers.drop('outlier' , axis=1)

            self.X = self.X.drop('outlier' , axis=1 , errors='ignore')
            self.y = self.y.drop('outlier' , axis=1 , errors='ignore')

            scores_without_outliers = np.mean(cross_val_score(estimator=self.best_model , X=X_without_outliers , y=y_without_outliers , cv=self.n_cv))
            scores_with_outliers = np.mean(cross_val_score(estimator=self.best_model , X=self.X , y=self.y.values , cv=self.n_cv))

            if scores_without_outliers > scores_with_outliers:
                self.outliers = 'Better get rid of the most extreme 5 percent'
                self.X = X_without_outliers
                self.y = y_without_outliers

            else:
                self.outliers = 'Better keep the outliers'


    def get_best_setup(self):
        self.chose_model()
        self.chose_best_scaler()
        self.chose_between_target_encoding_and_ohe()
        self.with_or_without_outliers()

        return f'Best model : {self.best_model}\nBest scaler : {self.best_scaler}\nBest categorical processing : {self.best_cat_prepro}\
        \nWith or without outliers : {self.outliers}'

    def get_X_and_y(self):
        self.X.drop('target' ,  axis=1 , inplace=True , errors='ignore')
        return self.X , self.y
