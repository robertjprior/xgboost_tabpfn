import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, \
    cross_val_score, ShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, \
    precision_score, recall_score
from sklearn.preprocessing import PolynomialFeatures
import optuna
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy


import matplotlib.pyplot as plt

from tabpfn import TabPFNClassifier
np.random.seed(37)
optuna.logging.set_verbosity(optuna.logging.WARNING)
use_optuna = True
#TODO:
#data inputation 
#encoding of objects - done
#interaction of variables - done
#FIXME:



#_______DataSetup___________
df = pd.read_excel('data/Breast Cancer Prediction_Datasets_training.xlsx')
df_test = pd.read_excel('data/Breast Cancer Prediction_Datasets_test.xlsx')

y = df['diagnosis']
y_test = df_test['diagnosis']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_test = label_encoder.transform(y_test)

df = df.drop('diagnosis', axis=1)
df_test = df_test.drop('diagnosis', axis=1)

results_report = open("reports/results_report.txt", "w")

#_______Functions______________

class pipeline_data_cleaning(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    def fit(self, X, y = None):
        #print(X.columns)
        X_ = self._universal_cleaning(X)
        return self
    
    def transform(self, X, y= None):
        X_ = self._universal_cleaning(X)
        return X_

    def _universal_cleaning(self, df):
        df = df.drop(['id','Unnamed: 32'],axis=1) #try it out, else use numpy version below
        #print(df.shape)
        #df = df.values[:, 0:-1]
        return df
        


def xgb_train(X_train, y_train, use_optuna=True):
    grid = None
    pipeline = None
    if use_optuna is True:
        X_tr, X_v, y_tr, y_v = train_test_split(X_train, y_train, test_size=0.3, \
            random_state=8)
        def get_model(params={}):
            #feature_selector = SelectKBest(f_classif, k=3)
            fs_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            boruta_features = BorutaPy(
                verbose=2,
                estimator=fs_model,
                n_estimators='auto',
                max_iter=10,  
                random_state=42,
            )
            scaler = StandardScaler()
            #imputer = SimpleImputer(strategy=)
            interaction_generator = PolynomialFeatures(degree = 1, \
                interaction_only=False, include_bias=False)
            model = xgb.XGBClassifier(eval_metric = 'logloss', use_label_encoder=False)
            data_cleaner = pipeline_data_cleaning()

            pipeline = Pipeline([
                ('data_cleaner', data_cleaner),
                ('standard_scaler', scaler), 
                ('interaction_gen', interaction_generator),
                ('feature_selection', boruta_features), 
                ('model', model)
            ])

            pipeline.set_params(**params)

            return pipeline

        def objective(trial):
            params = {
                'interaction_gen__interaction_only': trial.suggest_categorical('interaction_gen__interaction_only', [True, False]),
                #'feature_selection__k': trial.suggest_int('feature_selection__k', 10, 30),
                'model__max_depth': trial.suggest_int('model__max_depth', 2, 8), 
                'model__gamma': trial.suggest_float('model__gamma', 0, 0.5),
                'model__subsample': trial.suggest_float('model__subsample', 0.5, 1.0),
                'model__eval_metric': trial.suggest_categorical('model__eval_metric', ['merror', 'mlogloss']), 
                'model__learning_rate': trial.suggest_float('model__learning_rate', 0.1, 0.35), 
                'model__lambda': trial.suggest_float('model__lambda', 0.8, 1.0), 
                'model__scale_pos_weight': trial.suggest_float('model__scale_pos_weight', 0.6, 1.0), 
                'model__n_estimators': trial.suggest_int('model__max_depth', 20, 50), 
            }

            model = get_model(params)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_v)

            ps = precision_score(y_v, y_pred)
            rs = recall_score(y_v, y_pred)
            f1 = f1_score(y_v, y_pred)

            trial.set_user_attr('precision', ps)
            trial.set_user_attr('recall', rs)
            trial.set_user_attr('f1', f1)

            return f1


        study = optuna.create_study(**{
            'direction': 'maximize',
            'sampler': optuna.samplers.TPESampler(seed=37),
            'pruner': optuna.pruners.MedianPruner(n_warmup_steps=10)
        })

        study.optimize(**{
            'func': objective,
            'n_trials': 200,
            'n_jobs': -1,
            'show_progress_bar': True
        })

        #study.best_params
        #study.best_value
        grid = study
        pipeline = get_model(study.best_params)

    else:
        feature_selector = SelectKBest(f_classif, k=3)
        scaler = StandardScaler()
        #imputer = SimpleImputer(strategy=)
        interaction_generator = PolynomialFeatures(degree = 1, interaction_only=False, include_bias=False)
        model = xgb.XGBClassifier(eval_metric = 'logloss', use_label_encoder=False)
        data_cleaner = pipeline_data_cleaning()

        pipeline = Pipeline([
            ('data_cleaner', data_cleaner),
            ('standard_scaler', scaler), 
            ('interaction_gen', interaction_generator),
            ('feature_selection', feature_selector), 
            ('model', model)
        ])
        #np.logspace(5, 30, 5)
        param_grid = {
            'interaction_gen__interaction_only': [True, False],
            'feature_selection__k': [15,20,25],
            'model__max_depth': [2, 3, 5],
            'model__gamma': [0, 0.1, 0.2],
            'model__subsample': [0.5, 0.75, 1],
            'model__eval_metric': ['merror', 'mlogloss'],
            'model__learning_rate': [0.1, 0.15, 0.2, 0.25],
            'model__lambda': [1, 0.9],
            'model__scale_pos_weight': [1, 0.6],
            'model__n_estimators': [10,20,30],
            #'model__grow_policy': [0,1], #{'depthwise', 'lossguide'}
        }

        grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc', refit=True)
        grid.fit(X_train, y_train)

        results_report.write(str(grid.best_params_) + '\n')
        #print(grid.best_params_)
    return grid, pipeline



def predict(pipeline, x_df):
    return pipeline.predict(x_df)



#________PreModeling_________
#kf = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=8)
kf = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

#_____XGBoost____________
results_report.write('\n\n' + str("XGBoost") + '\n')
model, xg_pipeline = xgb_train(X_train, y_train, use_optuna=use_optuna)

#make new pipeline separate from gridsearchcv
if use_optuna is True:
    results_report.write(str(model.best_params) + '\n')
    #xg_pipeline.set_params(**model.best_params_) #already done in trainer
    xg_pipeline.fit(X_train, y_train)
else:
    results_report.write(str(model.best_params_) + '\n')
    xg_pipeline.set_params(**model.best_params_) 
    xg_pipeline.fit(X_train, y_train)


#training insample score
xgb_pred_train = predict(xg_pipeline, X_train)
results_report.write(str(classification_report(y_train, xgb_pred_train)) + '\n')
print(accuracy_score(y_train, xgb_pred_train)) 
print(classification_report(y_train, xgb_pred_train))

#validation out of sample score
xgb_pred_val = predict(xg_pipeline, X_val)
results_report.write(str(classification_report(y_val, xgb_pred_val)) + '\n')
print(accuracy_score(y_val, xgb_pred_val)) 
print(classification_report(y_val, xgb_pred_val))

#now cross val to triple check performance
xgb_scores = cross_validate(xg_pipeline, df, y, scoring="accuracy", return_train_score=True)
results_report.write('Accuracy Score:' + str(xgb_scores) + '\n')
print(xgb_scores)

xgb_scores = cross_validate(xg_pipeline, df, y, scoring="f1", return_train_score=True)
results_report.write('f1 score:' + str(xgb_scores) + '\n')
print(xgb_scores)

#refit on all the data - final model
xg_pipeline.fit(df, y)
xgb.plot_importance(xg_pipeline['model'])
plt.savefig('reports/feature_importance.png')


#test dataset
test_pred = xg_pipeline.predict(df_test)
results_report.write('final test accuracy:' + str(accuracy_score(y_test, test_pred)) + '\n')
results_report.write('final test f1 score:' + str(f1_score(y_test, test_pred)) + '\n')

#install graphviz for below
# xgb.plot_tree(xg_pipeline['model'], num_trees=1)
# plt.savefig('reports/tree1.png')

# xgb.plot_tree(xg_pipeline['model'], num_trees=2)
# plt.savefig('reports/tree2.png')
#xg_pipeline.fit(X_train, y_train, model__eval_set=[(X_train, y_train), (X_val, y_val)]) #model__evals_result=dict_eval)





#________TabPFN____________
results_report.write('\n\n' + str("TabPFN") + '\n')
classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

classifier.fit(X_train, y_train)

#training insample score
tabfpn_pred_train = classifier.predict(X_train)
results_report.write(str(classification_report(y_train, tabfpn_pred_train)) + '\n')
print(classification_report(y_train, tabfpn_pred_train))

#validation out of sample score
tabfpn_pred_val = classifier.predict(X_val)
results_report.write(str(classification_report(y_val, tabfpn_pred_val)) + '\n')
print(classification_report(y_val, tabfpn_pred_val))

#CV optimal model
tabpfn_scores = cross_validate(classifier, df, y, scoring="accuracy", return_train_score=True) #cv=kf, scoring="accuracy", return_train_score=True)
results_report.write(str(tabpfn_scores) + '\n')
print(tabpfn_scores) 



classifier.fit(df, y)

test_pred = classifier.predict(df_test)
results_report.write('final test accuracy:' + str(accuracy_score(y_test, test_pred)) + '\n')
results_report.write('final test f1 score:' + str(f1_score(y_test, test_pred)) + '\n')

results_report.close()

#from sklearn.compose import ColumnTransformer

#full_processor = ColumnTransformer(
#    transformers=[
#        ("numeric", numeric_pipeline, num_cols),
#        ("categorical", categorical_pipeline, cat_cols),
#    ]
#)






