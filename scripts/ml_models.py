def rf_model_train(X_train, y_train, version):
    #--- train models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state = 42))
    ])
    parameters = { 
        'clf__n_estimators': [10, 100],
        'clf__max_features': ['auto', 'sqrt', 'log2'],
        'clf__max_depth' : [10,100, 1000],
        #'clf__criterion' : ['gini', 'entropy'],
        'clf__class_weight' : [None, 'balanced', 'balanced_subsample']
    }
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(pipeline, param_grid = parameters, cv = 5, n_jobs=20, scoring="f1_weighted")
    model.fit(X_train, y_train)

    print("\nBest: %f using %s" % (model.best_score_, model.best_params_))
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("      %f (%f) with: %r" % (mean, stdev, param))

    #-- Save model
    from joblib import dump, load
    dump(model, '../models/%s_rf-model.joblib' %version)
    
    return model

def xgb_model_train(X_train, y_train, version):
    #--- train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    estimator = XGBClassifier(
        objective= 'multi:softmax',
        nthread=4,
        seed=42
    )
    parameters = {
        'max_depth': range (2, 10, 2),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05],
        
    }
    
    model = GridSearchCV(estimator, param_grid = parameters, cv = 5, n_jobs=20, scoring="f1_weighted")
    # weighting to handle imbalanced data
    from sklearn.utils import class_weight
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced', y=y_train)

    model.fit(X_train, y_train, sample_weight=classes_weights)
    #model.fit(X_train, y_train)

    print("\nBest: %f using %s" % (model.best_score_, model.best_params_))
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("      %f (%f) with: %r" % (mean, stdev, param))

    #-- Save model
    from joblib import dump, load
    dump(model, '../models/%s_xgb-model.joblib' %version)

    return model

def svc_model_train(X_train, y_train, version):
    from sklearn.svm import SVC 
    from sklearn.model_selection import GridSearchCV 
  
    # defining parameter range 
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf']}  
    
    model = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0, cv = 5, n_jobs=20, scoring="f1_weighted") 
    # fitting the model for grid search 
    model.fit(X_train, y_train)
    print("\nBest: %f using %s" % (model.best_score_, model.best_params_))
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("      %f (%f) with: %r" % (mean, stdev, param))

    #-- Save model
    from joblib import dump, load
    dump(model, '../models/%s_svc-model.joblib' %version)

    return model