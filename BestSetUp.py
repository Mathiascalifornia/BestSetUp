import pandas as pd , numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pyod.models.iforest import IForest
from sklearn.model_selection import cross_val_score , RepeatedStratifiedKFold
from sklearn.metrics import make_scorer , f1_score
from tqdm import tqdm


class BestSetUp:

    """
    This class aims to save time by providing the best options for a data science problem, including model selection, scaler/transformer selection, preprocessing of categorical data, evaluation of PCA, and managing outliers. All results are obtained using cross-validation.
    
    Attributes:
        X (pd.DataFrame): DataFrame of explanatory variables (with no missing values and unuseful features removed).
        y (pd.DataFrame): DataFrame of the target variable (with one column).
        models (list): A list of models that can function with a sklearn fashion.
        scalers (list): A list of sklearn scalers.
        n_cv (int): Number of cross-validations to perform at each step.
        multiclass (bool): Indicates if the problem is multiclass or not.
        n_components (float): Value representing the percentage of variance explained by PCA components (default is 0.95).
    
    Methods:
        chose_model(): Selects the best model for the dataset using MinMaxScaler and One-hot encoding as preprocessing steps.
        chose_best_scaler(): Selects the best scaler using only the numerical columns.
        chose_between_target_encoding_and_ohe(): Selects between one-hot encoding and target encoding for categorical data.
        with_or_without_outliers(): Compares scores between the dataframe without the 5 percent most extreme values and the original.
        get_best_setup(): Returns a string with the best setup found for the given problem.
        get_X_and_y(): Returns preprocessed X and y using the new setup.
        get_best_model(): Returns the best model.
        get_best_scaler(): Returns the best scaler.

    """

    def __init__(self , X : pd.DataFrame , y : pd.DataFrame , models : list , scalers : list , n_cv : int , multiclass : bool , n_componants=0.95):
        """
        Initializes the BestSetUp class with provided parameters.

        Args:
            X (pd.DataFrame): DataFrame of explanatory variables.
            y (pd.DataFrame): DataFrame of the target variable.
            models (list): A list of models.
            scalers (list): A list of scalers.
            n_cv (int): Number of cross-validations to perform.
            multiclass (bool): Indicates if the problem is multiclass or not.
            n_components (float, optional): Percentage of variance explained by PCA components. Defaults to 0.95.
        """
        self.X = X.copy()
        self.y = y.copy()

        self.models = models 
        self.scalers = scalers 
        self.n_cv = n_cv
        self.multiclass = multiclass
        self.cv_type = None
        self.with_pca = False
        self.n_componants = n_componants # Based value is 95 % of variance explained , but feel free to use another value
        self.outliers = None # In case it crashs


        if isinstance(self.y, pd.Series):
            self.y = pd.DataFrame(self.y)

        self.y.columns = ['target']

        if self.y['target'].nunique() == 2: # Unbalanced binary problem
            if float(self.y.sum()) / len(self.y) <= 0.30 or len(self.y) / float(self.y.sum()) <= 0.30: # Unbalanced class
                self.cv_type = RepeatedStratifiedKFold(n_splits=self.n_cv)


        if self.multiclass == True: # Multiclassification problem
            self.cv_type = RepeatedStratifiedKFold(n_splits=self.n_cv)
        
            

        if self.multiclass == False and self.y['target'].nunique() > 2: # For regression problems
            self.scoring = 'neg_root_mean_squared_error'
        else:
            self.scoring = make_scorer(f1_score) # For classification


    def chose_model(self):
        """
        Selects the best model for the dataset using MinMaxScaler and One-hot encoding as preprocessing steps.
        """       
        try:
            # Preprocess categorical data for testing purposes
            cat_cols = self.X.select_dtypes(include=object)    
            X_ = pd.concat([pd.get_dummies(cat_cols) , self.X.select_dtypes(include=np.number)] , axis=1)  
        except:
            X_ = self.X
            
        # Scale the numerical features using MinMaxScaler
        for col in X_.select_dtypes(include=np.number):
            if X_[col].nunique() >= 5: # To spare the categorical columns
                X_[col] = MinMaxScaler().fit_transform(X_[col].values.reshape(-1,1))


        best_model = self.models[0]
        best_score = np.mean(cross_val_score(estimator=best_model , X=X_, y=self.y, cv=self.n_cv, scoring=self.scoring))

 
        for model in tqdm(self.models):
                
                # Evaluate models
                if self.cv_type != None:

                    cv = self.cv_type
                    scores = cross_val_score(model, X_, self.y, cv=cv, scoring=self.scoring)
                
                else:
                    scores = cross_val_score(model, X_, self.y, cv=self.n_cv, scoring=self.scoring)
                
                # Mean score
                mean_score = np.mean(scores)
                
                # Check if the current best score is higher than the previous one
                if mean_score > best_score:
                    best_model = model
                    best_score = mean_score
            
        self.best_model = best_model




    def chose_best_scaler(self):
        """
        Selects the best scaler using only the numerical columns.
        """
        
        # Work only with the numerical features
        to_drop = []
        for col in self.X.select_dtypes(include=np.number):
            if self.X[col].nunique() <= 2: # Not binary
                to_drop.append(col)
        
        # Drop the ordinal / binary / columns with not enought diversity
        X_num = self.X.copy()
        X_num = X_num.select_dtypes(include=np.number).drop(to_drop , axis=1)


        best_scaler = self.scalers[0] 
        best_score = np.mean(cross_val_score(estimator=self.best_model , X=X_num , y=self.y , cv=self.n_cv , scoring=self.scoring)) # Minimize


        for scaler in tqdm(self.scalers):
            X_num_copy = X_num.copy()
            X_num_copy[X_num_copy.columns] = scaler.fit_transform(X_num_copy[X_num_copy.columns])

            # Evaluate models
            if self.cv_type != None:

                cv = self.cv_type
                scores = cross_val_score(self.best_model, X=X_num_copy, y=self.y, cv=cv, scoring=self.scoring)
                
            else:

                scores = cross_val_score(self.best_model, X=X_num_copy, y=self.y, cv=self.n_cv, scoring=self.scoring)


            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_scaler = scaler
                best_score = mean_score

        self.best_scaler = best_scaler
        self.X[X_num.columns] = best_scaler.fit_transform(self.X[X_num.columns])
    


    def with_or_without_pca(self):
        """
        Apply Principal Component Analysis (PCA) and determine whether to include PCA-transformed features or not.

        The method compares the mean cross-validation scores of the best model using the dataset with PCA-transformed features and without PCA.
        If the score with PCA is higher or equal to the score without PCA, the method updates the feature matrix X by adding the PCA-transformed features and dropping the original numerical features.

        Note: This method assumes that self.X and self.y have been preprocessed and self.best_model has been defined.
        """
        X_num = self.X.select_dtypes(include=np.number).copy()
        pca = PCA(n_components=self.n_componants)
        pca.fit(X_num)
        pca_columns = [f'pca_{i}' for i in range(pca.n_components_)]
        pca_X = pd.DataFrame(pca.transform(X_num) , columns=pca_columns)

        score_with_pca = np.mean(cross_val_score(self.best_model, X=pca_X, y=self.y, cv=self.n_cv, scoring=self.scoring))
        score_without_pca = np.mean(cross_val_score(self.best_model, X=X_num, y=self.y, cv=self.n_cv, scoring=self.scoring))

        if score_with_pca >= score_without_pca:
            self.X[pca_columns] = pca_X
            self.X = self.X.drop(X_num.columns , axis=1 , errors='ignore')
            self.with_pca = True






    def chose_between_target_encoding_and_ohe(self):
            """
            Selects between one-hot encoding and target encoding for categorical data.
            """
            # Transform X using the best scaler
            for col in self.X.select_dtypes(include=np.number):
                if self.X[col].nunique() <= 5:
                    self.X[col] = self.best_scaler.fit_transform(self.X[col].values.reshape(-1,1))

            # X one hot encode
            dummies = pd.get_dummies(self.X.select_dtypes(include=object))
            X_ohe = pd.concat([self.X , dummies] , axis='columns')
            X_ohe = X_ohe.drop(list(X_ohe.select_dtypes(include=object).columns) , axis=1 , errors='ignore')
            X_ohe['target'] = self.y.values


            if self.scoring == make_scorer(f1_score) or self.multiclass == True or self.y['target'].nunique() <= 5:
                self.best_cat_prepro = 'One-Hot-Encoding' # For binary or label encoding it should be done with domain knowledge , before using this class
                self.X = X_ohe
            

            else: # Because target encoding works only for regression problems

                X_te = self.X.copy() # The target encoding dataframe
                X_te['target'] = self.y.values

                for col in X_te.select_dtypes(include=object):
                    means_values = dict(X_te.groupby(col)['target'].mean())
                    X_te[col] = [means_values.get(val) for val in list(X_te[col])]
                    X_te[col] = self.best_scaler.fit_transform(X_te[col].values.reshape(-1 , 1))

                
                target_encoding_score = np.mean(cross_val_score(estimator=self.best_model , X=X_te.drop('target' , axis=1) , y=self.y , cv=self.n_cv , n_jobs=-1))
                ohe_score = np.mean(cross_val_score(estimator=self.best_model , X=X_ohe.drop('target' , axis=1) , y=self.y , cv=self.n_cv , n_jobs=-1))


                if target_encoding_score > ohe_score:
                    self.X = X_te
                    self.best_cat_prepro = 'Target encoding'
                    
                else:
                    self.X = X_ohe
                    self.best_cat_prepro = 'One-Hot_Encoding'

        


    def with_or_without_outliers(self):
            """
            Compares scores between the dataframe without the 5 percent most extreme values and the original.
            """           
            iforest = IForest(contamination=0.05) # Drop the most extreme 5 per cent of the dataframe
            iforest.fit(self.X)
            is_outliers = list(iforest.predict(self.X))


            self.X['outlier'] = is_outliers
            self.y['outlier'] = is_outliers
            
            X_without_outliers = self.X[self.X['outlier'] != 1]
            y_without_outliers = self.y[self.y['outlier'] != 1]

            X_without_outliers = X_without_outliers.drop('outlier' , axis=1)
            y_without_outliers = y_without_outliers.drop('outlier' , axis=1)


            self.X = self.X.drop('outlier' , axis=1)
            self.y = self.y.drop('outlier' , axis=1)

            if self.cv_type != None:
                cv = self.cv_type
                scores_without_outliers = np.mean(cross_val_score(estimator=self.best_model , X=X_without_outliers , y=y_without_outliers , cv=cv))
                scores_with_outliers = np.mean(cross_val_score(estimator=self.best_model , X=self.X , y=self.y.values , cv=cv))

            else:
                scores_without_outliers = np.mean(cross_val_score(estimator=self.best_model , X=X_without_outliers , y=y_without_outliers , cv=self.n_cv))
                scores_with_outliers = np.mean(cross_val_score(estimator=self.best_model , X=self.X , y=self.y.values , cv=self.n_cv))

            if scores_without_outliers > scores_with_outliers:
                self.outliers = 'Better get rid of the most extreme 5 percent'
                self.X = X_without_outliers
                self.y = y_without_outliers

            else:
                self.outliers = 'Better keep the outliers'

            



    def get_best_setup(self):
        """
        Returns a string with the best setup found for the given problem.

        Returns:
            str: String with the best model, best scaler, PCA usage, categorical processing, and outliers handling.
        """

        print('Choosing model ...')
        self.chose_model()
        print('Choosing scaler ...')
        self.chose_best_scaler()
        print('Trying PCA ...')
        self.with_or_without_pca()
        print('Choosing categorical preprocessing ...')
        try:
            self.chose_between_target_encoding_and_ohe()
        except:
            print('No categorical data to deal with')
            self.best_cat_prepro = None
        print('Dealing with outliers ...\n')
        try:
            self.with_or_without_outliers()
        except:
            pass

        return f'Best model : {self.best_model}\nBest scaler : {self.best_scaler}\nWith or without PCA : {self.with_pca}\nBest categorical processing : {self.best_cat_prepro}\
        \nWith or without outliers : {self.outliers}'

    def get_X_and_y(self):
        """
        Get preprocessed X and y data.

        Returns:
            X (pd.DataFrame): Preprocessed feature X.
            y (pd.DataFrame): Preprocessed target variable y.
        """
        self.X = self.X.drop('target' ,  axis=1 , errors='ignore')
        self.X = self.X.drop('outlier' , axis=1 , errors='ignore')
        return self.X , self.y
    
    def get_best_model(self): 
        """
        Get the best model identified.

        Returns:
            best_model: The best model identified for the given dataset.
        """
        return self.best_model

    def get_best_scaler(self):
        """
        Get the best scaler identified.

        Returns:
            best_scaler: The best scaler identified after fitting on the X data.
        """
        self.best_scaler.fit(self.X)
        return self.best_scaler