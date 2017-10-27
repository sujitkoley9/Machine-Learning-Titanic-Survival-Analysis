#----------------- Step 1:  Importing required packages for this problem ------------------------------------- 
   # data analysis and wrangling
    import pandas as pd
    import numpy as np
    import random as rn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt

    # machine learning
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import cross_val_score
    
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    from xgboost.sklearn import XGBRegressor
    from xgboost  import plot_importance
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    
    
#--------- Step 2:  Reading and loading train and test datasets and generate data quality report---------------- 
      
    # loading train and test sets with pandas 
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    #append two train  and test dataframe
    full  = train_df.append(test_df,ignore_index=True)
    
    # Print the columns of dataframe
    print(full.columns.values)
    
    # Returns first n rows
    full.head(10)
    
    
    # Retrive data type of object and no. of non-null object
    full.info()
    
    # Retrive details of integer and float data type 
    full.describe()
    
    # To get  details of the categorical types
    full.describe(include=['O'])

   

  ##Prepare data quality report-
  # To get count of no. of NULL for each data type columns = full.columns.values
    columns = full.columns.values
    data_types = pd.DataFrame(full.dtypes, columns=['data types'])
    
    missing_data_counts = pd.DataFrame(full.isnull().sum(),
                            columns=['Missing Values'])
    
    present_data_counts = pd.DataFrame(full.count(), columns=['Present Values'])
    
    UniqueValues = pd.DataFrame(full.nunique(), columns=['Unique Values'])
    
    MinimumValues = pd.DataFrame(columns=['Minimum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) | (full[c].dtypes == 'int64'):
            MinimumValues.loc[c]=full[c].min()
       else:
            MinimumValues.loc[c]=0
 
    MaximumValues = pd.DataFrame(columns=['Maximum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) |(full[c].dtypes == 'int64'):
            MaximumValues.loc[c]=full[c].max()
       else:
            MaximumValues.loc[c]=0
    
    data_quality_report=data_types.join(missing_data_counts).join(present_data_counts).join(UniqueValues).join(MinimumValues).join(MaximumValues)
    data_quality_report.to_csv('Titanic_data_quality.csv', index=True)


#------------------------------------Step 3:Feature Engineering----------------------------------------------
   # Here, we are creating a new col named title
   full['Title']=full.Name.str.extract(' ([A-Za-z]+)\.', expand=True )   
   full.loc[(full['Title'] != 'Mr' ) &(full['Title'] != 'Mrs' )&
        (full['Title'] != 'Master' ) &(full['Title'] != 'Miss' ),'Title'] ='RareTitle'
   
   
   
#-----------------------------------Step 4: Missing value treatment---------------------------------------------  
    # Acoording to report age and cabin has missing value
  
   # Replace mean value according to Title
   full['Age'].fillna(full.groupby('Title')['Age'].transform("mean"), inplace=True) 
   
   
    # Treatment for cabin  
  full['Cabin'].fillna('Missing', inplace=True)
  
    # Treatment for Fare  
  full['Fare'].fillna(full['Fare'].mean(), inplace=True)
        
   #Treatment for Embarked
   full['Embarked'].fillna('Missing', inplace=True)
   data1=full[0:891][['Survived','Embarked','Title']]. groupby(['Survived','Embarked'],as_index=False).count().\
         rename(columns={'Title':'Count'})
   data1=data1[data1['Survived']==1].reset_index()
   
   data2=full[0:891][['Embarked','Title']]. groupby(['Embarked'],as_index=False).count().\
         rename(columns={'Title':'Count'})
         
   data2=data2[data2['Embarked'].isin(data1['Embarked'])]
   
   Survived_percentage = round(data1['Count']/data2['Count'],3)
   
   summary =pd.DataFrame({'Embarked':data1['Embarked'],
                          'Survived_percentage':Survived_percentage,
                          'N1':data1['Count'],
                          'N2':data2['Count']})
                        
   
   #As per analysis Embarked Missing should be replaced by "C"
   full.loc[full['Embarked'] == 'Missing','Embarked']='C'
   
  
#-------------------------------------Step 5: Outlier Treatment -------------------------------------------  

# here is no need operation for outlier using BoxPlot.
    #BoxPlot=boxplot(full['Age'])
    #outlier= BoxPlot['fliers'][0].get_data()[1]
    #full.loc[full['Age'].isin(outlier),'Age']=full['Age'].mean()


#--------------------------------Step 6: Exploration analysis of data--------------------------------------

# Percent of male/female of survivals
    full[0:891][['Sex','Survived']].groupby('Sex',as_index=False).mean().sort_values(by='Survived', ascending=False)
    g= sns.factorplot(x='Sex', 
                   data=full[0:891], 
                   hue='Sex',  # Color by Sex
                   col='Survived' ,# Separate by Survived
                   kind='count') # barplot
                    
# Percent of each class survivals
    full[0:891][['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass', ascending=True)
    g= sns.factorplot(x='Pclass', 
                   data=full[0:891], 
                   hue='Pclass',  # Color by Plclass
                   col='Survived' ,# Separate by Survived
                   kind='count') # barplot

# Percent of survivals, linked to the fare
        full[0:891][['Fare','Survived']].groupby('Fare',as_index=False).mean().sort_values(by='Fare', ascending=False)
    
        grid = sns.FacetGrid(full[0:891], col='Survived', size=2.2, aspect=1.6)
        grid.map(sns.distplot, 'Fare')
        
# Percent of survivals, linked to the Title
        full[0:891][['Title','Survived']].groupby('Title',as_index=False).mean().sort_values(by='Survived', ascending=False)
        g= sns.factorplot(x='Title', 
                   data=full[0:891], 
                   hue='Title',  # Color by Sex
                   col='Survived' ,# Separate by Survived
                   kind='count') # barplot



#------------------------------------Step 7:Feature Engineering-----------------------------------------------
   # Creating Child Variable         
   full['Child'] =np.where(full['Age'] <=15, 1,0)
   
   # Change Sex variable as NUmeric 1/0
   full['Sex'] =np.where(full['Sex'] =='male', 1,0)
   
   # Create Family size variable
   full['FamilySize'] = full.SibSp + full.Parch +1
   
   # Creating Mother Variable         
   full['Mother'] =np.where((full['Title'] =='Mrs') & (full['Parch']>0) , 1,0)
   
   #Creating dummy variable for title using get_dummies
   Title_dummies = pd.get_dummies(full['Title'],prefix='Title')
   Title_dummies=Title_dummies.iloc[:,1:]
   full=full.join(Title_dummies)
    #Creating dummy variable for Embarked using get_dummies
   Embarked_dummies = pd.get_dummies(full['Embarked'],prefix='Embarked')
   Embarked_dummies=Embarked_dummies.iloc[:,1:]
   full=full.join(Embarked_dummies)
   
    #---------------------------------- Droping unnecessary columns-------------------------------
    full.drop(['Cabin','Name','PassengerId','Ticket','Title','Embarked'], axis=1, inplace=True)
    full.columns.values

   
   
  
#----------------------Step 8: Separating train/test dataset and Normalize data--------------------------------   
    train_new=full[0:891]
    test_new=full[891:]
    
    X_train = train_new.drop(['Survived'], axis=1)
    Y_train = train_new["Survived"]
    
    X_test  = test_new.drop(['Survived'], axis=1)
   
    #-----Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
#--------------------PCA to reduce dimension and remove correlation----------------------------    
     pca = PCA(n_components =15)
     pca.fit_transform(X_train)
     #The amount of variance that each PC explains
     var= pca.explained_variance_ratio_
     #Cumulative Variance explains
     var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
     plt.plot(var1)
     
     # As per analysis, we can skip 4 principal componet, use only 11 components
     
     pca = PCA(n_components =11)
     X_train=pca.fit_transform(X_train)
     X_test=pca.fit_transform(X_test)
     
     
#----------------------Step 9:Run Algorithm----------------------------------------------------------------------

   #1.Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    logreg_acc = cross_val_score(estimator = logreg, X = X_train, y = Y_train, cv =    10)
    logreg_acc_mean = logreg_acc.mean()
    logreg_std = logreg_acc.std()
    


   #2.Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    decision_tree_acc = cross_val_score(estimator = decision_tree, X = X_train, y = Y_train, cv =    10)
    decision_tree_acc_mean = decision_tree_acc.mean()
    decision_tree_std = decision_tree_acc.std()
    
    # Choose some parameter combinations to try
    parameters = {
                  'criterion': ['entropy', 'gini'],
                  'max_depth': range(2,10), 
                  'min_samples_split': range(2,10),
                  'min_samples_leaf': range(1,10)
                 }

    # Search for best parameters
    grid_obj = GridSearchCV(estimator=decision_tree, 
                                    param_grid= parameters,
                                    scoring = 'accuracy',
                                    cv = 10,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    decision_tree_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    decision_tree_best.fit(X_train, Y_train)
    
    # Calculate accuracy of decisison tree again
    decision_tree_acc = cross_val_score(estimator = decision_tree_best, X = X_train, y = Y_train, cv =    10)
    decision_tree_acc_mean = decision_tree_acc.mean()
    decision_tree_std = decision_tree_acc.std()
    #----------------------------------------------
     #---To Know importanve of variable
    feature_importance = pd.Series(decision_tree_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    
    
    
   #3.Random Forest
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, Y_train)
    random_forest_acc = cross_val_score(estimator = random_forest, X = X_train, y = Y_train, cv =    10)
    random_forest_acc_mean = random_forest_acc.mean()
    random_forest_std = random_forest_acc.std()



    # Choose some parameter combinations to try
    parameters = {'n_estimators': [20,30,40], 
                  'max_features': ['log2', 'sqrt','auto'], 
                  'criterion': ['entropy', 'gini'],
                  'max_depth': range(2,10), 
                  'min_samples_split': range(2,10),
                  'min_samples_leaf': range(1,10)
                 }
   
    grid_obj = GridSearchCV(estimator=random_forest, 
                                    param_grid= parameters,
                                    scoring = 'accuracy',
                                    cv = 10,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)
    

    
   

    # Set the clf to the best combination of parameters
    random_forest_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    random_forest_best.fit(X_train, Y_train)
    random_forest_acc = cross_val_score(estimator = random_forest_best, X = X_train, y = Y_train, cv =    10)
    random_forest_acc_mean = random_forest_acc.mean()
    random_forest_std = random_forest_acc.std()
    
    #---To Know importanve of variable
    feature_importance = pd.Series(random_forest_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    

   #4.XGBOOST
    Xgboost = XGBClassifier()
    Xgboost.fit(X_train, Y_train)
    Xgboost_acc = cross_val_score(estimator = Xgboost, X = X_train, y = Y_train, cv =    10)
    Xgboost_acc_mean = Xgboost_acc.mean()
    Xgboost_std = Xgboost_acc.std()



    # Choose some parameter combinations to try
   parameters = {'learning_rate':np.arange(0.1, .5, 0.1),
                  'n_estimators':[1000],
                  'max_depth': range(4,10),
                  'min_child_weight':range(1,5),
                  'reg_lambda':np.arange(0.55, .9, 0.05),
                  'subsample':np.arange(0.1, 1, 0.1),
                  'colsample_bytree':np.arange(0.1, 1, 0.1)
               }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=Xgboost, 
                                  param_distributions = parameters,
                                  scoring = 'accuracy',
                                  cv = 10,n_iter=300,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    Xgboost_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    Xgboost_best.fit(X_train, Y_train)
    Xgboost_acc = cross_val_score(estimator = Xgboost_best, X = X_train, y = Y_train, cv =    10)
    Xgboost_acc_mean = Xgboost_acc.mean()
    Xgboost_std = Xgboost_acc.std()
    
     #---To Know importanve of variable
    plot_importance(Xgboost_best)
    pyplot.show()
    
   #5.SVM
    SVM_Classifier=SVC()
    SVM_Classifier.fit(X_train, Y_train)
    SVM_Classifier_acc = cross_val_score(estimator = SVM_Classifier, X = X_train, y = Y_train, cv =    10)
    SVM_Classifier_acc_mean = SVM_Classifier_acc.mean()
    SVM_Classifier_std = SVM_Classifier_acc.std()



    # Choose some parameter combinations to try
   parameters = { 'kernel':('linear', 'rbf'),
                  'gamma': [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5],
                  'C': np.arange(1, 10,1)
                 }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=SVM_Classifier, 
                                  param_distributions = parameters,
                                  scoring = 'accuracy',
                                  cv = 3,n_iter=100,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    SVM_Classifier_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    SVM_Classifier_best.fit(X_train, Y_train)
    SVM_Classifier_acc = cross_val_score(estimator = SVM_Classifier_best, X = X_train, y = Y_train, cv =    10)
    SVM_Classifier_acc_mean = SVM_Classifier_acc.mean()
    SVM_Classifier_std = SVM_Classifier_acc.std()
  

   #.6.KNN
    KNN_Classifier=KNeighborsClassifier() 
    KNN_Classifier.fit(X_train, Y_train)
    KNN_Classifier_acc = cross_val_score(estimator = KNN_Classifier, X = X_train, y = Y_train, cv =    10)
    KNN_Classifier_acc_mean = KNN_Classifier_acc.mean()
    KNN_Classifier_std = KNN_Classifier_acc.std()



    # Choose some parameter combinations to try
   parameters = { 'n_neighbors': np.arange(1, 31, 2),
	              'metric': ["euclidean", "cityblock"]
                 }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=KNN_Classifier, 
                                  param_distributions = parameters,
                                  scoring = 'accuracy',
                                  cv = 10,n_iter=30,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    KNN_Classifier_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    KNN_Classifier_best.fit(X_train, Y_train)
    KNN_Classifier_acc = cross_val_score(estimator = KNN_Classifier_best, X = X_train, y = Y_train, cv =    10)
    KNN_Classifier_acc_mean = KNN_Classifier_acc.mean()
    KNN_Classifier_std = KNN_Classifier_acc.std()
    
    
    
  #7.Artificial Neural network    

   def build_classifier(optimizer):
        ANN_classifier = Sequential()
        ANN_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
        ANN_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        ANN_classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        ANN_classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return ANN_classifier
    
    classifier = KerasClassifier(build_fn = build_classifier)
   
    # Choose some parameter combinations to try
    parameters = {'batch_size': [25, 32],
                  'epochs': [10, 20],
                  'optimizer': ['adam', 'rmsprop']
                  }
    
    # Search for best parameters
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 3)
    
    grid_search = grid_search.fit(X_train, Y_train)
   
 
    # Set the clf to the best combination of parameters
    ANN_Classifier_best = grid_search.best_estimator_
    
     # Fit the best algorithm to the data. 
    ANN_Classifier_best.fit(X_train, Y_train)
    ANN_Classifier_acc = cross_val_score(estimator = ANN_Classifier_best, X = X_train, y = Y_train, cv =    10)
    ANN_Classifier_acc_mean = ANN_Classifier_acc.mean()
    ANN_Classifier_std = ANN_Classifier_acc.std()
    

 #---------------Step 10:Prediction on test data ----------------------------------------------------------------
     Y_pred1 = logreg.predict(X_test)
     Y_pred2 = decision_tree_best.predict(X_test)
     Y_pred3 = random_forest_best.predict(X_test)
     Y_pred4 = Xgboost_best.predict(X_test)
     Y_pred5 = SVM_Classifier_best.predict(X_test)
     Y_pred6 = KNN_Classifier_best.predict(X_test)
     Y_pred7 = ANN_Classifier_best.predict(X_test)
     Y_pred7=Y_pred7.flatten()
     
     Y_pred = np.where((Y_pred1+Y_pred2+Y_pred3+Y_pred4+Y_pred5+Y_pred6+Y_pred7)/7>0.5,1,0)
     
    
     
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred1
        })
    submission.to_csv('submission.csv', index=False)





 
 