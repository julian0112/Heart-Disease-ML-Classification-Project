import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, roc_auc_score, make_scorer, f1_score
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('heart-disease.csv')
rs = 123
results = []

def grid_search_lr(X_train, y_train):
    params_grid = {
    'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}, {0:0.3, 1:0.7}, {0:0.4, 1:0.6}, {0:0.5, 1:0.5},]
    }
    lr_model = LogisticRegression(random_state=rs, max_iter=1000)
    grid_search = GridSearchCV(estimator = lr_model, 
                           param_grid = params_grid, 
                           scoring='f1',
                           cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def grid_search_rf(X_train, y_train):
    params_grid = {
        'max_depth': [5, 10, 15, 20],
        'n_estimators': [25, 50, 100],
        'min_samples_split': [2, 5],
        'class_weight': [{0:0.1, 1:0.9}, {0:0.2, 1:0.8}, {0:0.3, 1:0.7}, {0:0.4, 1:0.6}, {0:0.5, 1:0.5}]
    }
    rf_model = RandomForestClassifier(random_state=rs)
    grid_search = GridSearchCV(estimator = rf_model, 
                           param_grid = params_grid, 
                           scoring='f1',
                           cv = 5, verbose = 1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def grid_search_knn(X_train, y_train):
    params_grid = {
        'n_neighbors': [2, 3, 4, 5],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto'],
        'leaf_size': [25, 30, 40],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn_model,
                               param_grid=params_grid,
                               scoring='f1',
                               cv = 5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def grid_seacrh_lsvc(X_train, y_train):
    params_grid = {
        'penalty': ['l1', 'l2'],
        'C': [1, 3, 5, 7, 9, 10, 12, 15],
        'class_weight': [{0:0.1, 1:0.9}, {0:0.2, 1:0.8}, {0:0.3, 1:0.7}, {0:0.4, 1:0.6}, {0:0.5, 1:0.5}],
    }
    
    lscv_model = LinearSVC(random_state=rs, max_iter=2000)
    grid_search = GridSearchCV(estimator=lscv_model,
                                    param_grid=params_grid,
                                    scoring='f1',
                                    cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def evaluate(y_test, preds):
    precision, recall, f_beta, support = precision_recall_fscore_support(y_test, preds, beta=5, pos_label=1, average='binary')
    auc = roc_auc_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy is: {accuracy:.2f}")
    print(f"Precision is: {precision:.2f}")
    print(f"Recall is: {recall:.2f}")
    print(f"Fscore is: {f_beta:.2f}")
    print(f"AUC is: {auc:.2f}")

def resample(X_train, y_train):
    # SMOTE sampler (Oversampling)
    smote_sampler = SMOTE(random_state = 123)
    # Undersampling
    under_sampler = RandomUnderSampler(random_state=123)
    # Resampled datasets
    X_smo, y_smo = smote_sampler.fit_resample(X_train, y_train)
    X_under, y_under = under_sampler.fit_resample(X_train, y_train)
    return X_smo, y_smo, X_under, y_under

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_smo, y_smo, X_under, y_under = resample(X_train, y_train)

my_palette = {}

plt.figure()
sns.kdeplot(data=data, x='age', hue='sex')
plt.show()

# print(grid_search_lr(X_under, y_under))
# print(grid_search_rf(X_under, y_under))
# print(grid_search_knn(X_under, y_under))
# print(grid_seacrh_lsvc(X_under, y_under))

# print(y_smo.value_counts())

# lr_model = LogisticRegression(random_state = rs, max_iter = 1000, class_weight = {0: 0.4, 1: 0.6}, penalty = 'l2')
# lr_model.fit(X_train, y_train)
# preds = lr_model.predict(X_test)

# print('LOGISTIC REGRESSION WEIGHTED')
# evaluate(y_test, preds)

# lr_model_smo = LogisticRegression(random_state = rs, max_iter = 1000, class_weight = {0: 0.4, 1: 0.6}, penalty = 'l2')
# lr_model_smo.fit(X_smo, y_smo)
# preds_smo = lr_model_smo.predict(X_test)

# print()
# print('LOGISTIC REGRESSION SMOTE WEIGHTED')
# evaluate(y_test, preds_smo)

# lr_model_und = LogisticRegression(random_state = rs, max_iter = 1000, class_weight = {0: 0.5, 1: 0.5}, penalty = 'l2')
# lr_model_und.fit(X_under, y_under)
# preds_under = lr_model_und.predict(X_test)

# print()
# print('LOGISTIC REGRESSION UNDERSAMPLING WEIGHTED')
# evaluate(y_test, preds_under)

# rf_model = RandomForestClassifier(random_state = rs, bootstrap=True, class_weight={0: 0.2, 1: 0.8}, max_depth=5, min_samples_split=5, n_estimators=50)
# rf_model = rf_model.fit(X_train, y_train)
# preds = rf_model.predict(X_test)

# print('RANDOM FOREST WEIGHTED')
# evaluate(y_test, preds)

# rf_model_smo = RandomForestClassifier(random_state = rs, bootstrap=True, class_weight={0: 0.4, 1: 0.6}, max_depth=10, min_samples_split=2, n_estimators=50)
# rf_model_smo = rf_model_smo.fit(X_smo, y_smo)
# preds_smo = rf_model_smo.predict(X_test)

# print()
# print('RANDOM FOREST SMOTE WEIGHTED')
# evaluate(y_test, preds_smo)

# rf_model_under = RandomForestClassifier(random_state = rs, bootstrap=True, class_weight={0: 0.3, 1: 0.7}, max_depth=10, min_samples_split=5, n_estimators=50)
# rf_model_under = rf_model_under.fit(X_under, y_under)
# preds_under = rf_model_under.predict(X_test)

# print()
# print('RANDOM FOREST UNDERSAMPLING WEIGHTED')
# evaluate(y_test, preds_under)

# knn_model = KNeighborsClassifier(algorithm='auto', leaf_size=25, metric='manhattan', n_neighbors=5, p=1, weights='uniform')
# knn_model = knn_model.fit(X_train, y_train)
# preds_knn = knn_model.predict(X_test)

# print()
# print('KNN')
# evaluate(y_test, preds_knn)

# knn_model_smo = KNeighborsClassifier(algorithm='auto', leaf_size=25, metric='manhattan', n_neighbors=3, p=1, weights='distance')
# knn_model_smo = knn_model_smo.fit(X_smo, y_smo)
# preds_knn_smo = knn_model_smo.predict(X_test)

# print()
# print('KNN SMOTE')
# evaluate(y_test, preds_knn_smo)

# knn_model_under = KNeighborsClassifier(algorithm='auto', leaf_size=25, metric='manhattan', n_neighbors=5, p=1, weights='uniform')
# knn_model_under = knn_model_under.fit(X_under, y_under)
# preds_knn_under = knn_model_under.predict(X_test)

# print()
# print('KNN UNDERSAMPLING')
# evaluate(y_test, preds_knn_under)

# lscv_model = LinearSVC(C=3, class_weight={0: 0.3, 1: 0.7}, penalty='l2')
# lscv_model = lscv_model.fit(X_train, y_train)
# preds_lscv = lscv_model.predict(X_test)

# print()
# print('LINEAR SCV WEIGHTED')
# evaluate(y_test, preds_lscv)

# lscv_model = LinearSVC(C=1, class_weight={0: 0.5, 1: 0.5}, penalty='l1')
# lscv_model = lscv_model.fit(X_train, y_train)
# preds_lscv = lscv_model.predict(X_test)

# print()
# print('LINEAR SCV SMOTE WEIGHTED')
# evaluate(y_test, preds_lscv)

# lscv_model = LinearSVC(C=1, class_weight={0: 0.4, 1: 0.6}, penalty='l1')
# lscv_model = lscv_model.fit(X_train, y_train)
# preds_lscv = lscv_model.predict(X_test)

# print()
# print('LINEAR SCV UNDERSAMPLING WEIGHTED')
# evaluate(y_test, preds_lscv)