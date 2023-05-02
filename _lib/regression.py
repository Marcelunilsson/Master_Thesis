# %%
import os
import pickle as pkl
import time
from xgboost import XGBRegressor

import autokeras as ak
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
# import shutil


class Regressor:
    def __init__(self, X, y):
        self.y = y
        self.X = X
        self.features = len(X.columns)

    def set_model(self, model_name):
        self.model_name = model_name
        self.load_model()
        return self

    def set_max_trials(self, max_trials):
        self.max_trials = max_trials
        return self

    def set_data(self, X, y):
        self.y = y
        self.X = X
        return self

    def load_model(self):
        if self.model_name == 'ridge':
            self.model = Ridge()
            self.model.verbose = 1
            return self
        elif self.model_name == 'random_forest':
            self.model = RandomForestRegressor(
                verbose=1,
                n_jobs=-1,
                )
            return self
        elif self.model_name == 'xgboost':
            self.model = XGBRegressor()
            return self
        elif self.model_name == 'ak_struct_reg':
            self.model = ak.StructuredDataRegressor(
                overwrite=True,
                max_trials=self.max_trials,
                loss='mse',
                metrics=['mse'],
                directory='ak_struct_reg',
                project_name='ak_struct_reg',
                seed=42,
                tuner='bayesian',
                objective='val_loss',
                max_model_size=None)
            return self
        else:
            print('Model not found')
            return self

    def fit(self):
        if self.model_name == 'ak_struct_reg':
            self.model.fit(
                self.X.to_numpy(),
                self.y.to_numpy(),
                validation_split=0.2,
                # epochs=100, unspecified for adaptive epochs
                verbose=1)
        else:
            self.model.fit(
                self.X,
                self.y,
                validation_split=0.2,
                random_state=42)
        return self

    def export_model(self):
        return self.model.export_model()

    def predict(self, X, y):
        if self.model_name == 'ak_struct_reg':
            y_pred = self.model.predict(X.to_numpy()).flatten()
        else:
            y_pred = self.model.predict(X)
        performance = {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'r': np.corrcoef(y_pred, y)[0, 1],
            'model': self.model_name,
            }
        return performance, y_pred

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.model_name == 'ak_struct_reg':
            self.model.export_model(path)
        else:
            with open(path, 'wb') as f:
                pkl.dump(self.model, f)

    def load_model_from_path(self, path):
        if self.model_name == 'ak_struct_reg':
            self.model = ak.StructuredDataRegressor.load_model(path)
        else:
            with open(path, 'rb') as f:
                self.model = pkl.load(f)


class RegressorCV:
    def __init__(self, n_folds, X, y, embs_list=None):
        self.n_folds = n_folds
        self.set_data(X, y, embs_list)
        self.features = len(X.columns)

    def set_n_folds(self, n_folds):
        self.n_folds = n_folds
        return self

    def set_model(self, model_name):
        self.model_name = model_name
        self.load_model()
        return self

    def set_data(self, X, y, embs_list=None):
        self.y = y
        self.X = X
        self.embs_list = embs_list
        return self

    def scale_X(self, data, train=False):
        if train:
            self.scaler = StandardScaler()
            return pd.DataFrame(self.scaler.fit_transform(data))
        else:
            return pd.DataFrame(self.scaler.transform(data))

    def PCA_X(self, X, pca_n, train=False):
        if train:
            self.pca_list = [
                PCA(n_components=pca_n)
                for _ in range(len(self.embs_list))]
            X_list = [
                pd.DataFrame(pca.fit_transform(X.filter(regex=emb)))
                for pca, emb in zip(self.pca_list, self.embs_list)]
            return pd.concat(X_list, axis=1)
        else:
            X_list = [
                pd.DataFrame(pca.transform(X.filter(regex=emb)))
                for pca, emb in zip(self.pca_list, self.embs_list)]
            return pd.concat(X_list, axis=1)

    def bay_opt(self, param_space):
        bayes_search = BayesSearchCV(
            self.model,
            param_space,
            n_iter=50,
            cv=self.n_folds,
            n_jobs=-1,
            random_state=42,
            verbose=0)
        bayes_search.fit(self.X, self.y)
        return bayes_search.best_params_

    def load_model(self):
        if self.model_name not in ['ak_struct_reg']:
            param_space = self.get_param_space()
        if self.model_name == 'ridge':
            self.model = Ridge()
            self.model.verbose = 1
            params = self.bay_opt(param_space)
            # Use the best hyperparameters found
            self.model.set_params(**params)
            return self

        elif self.model_name == 'random_forest':
            self.model = RandomForestRegressor(verbose=1, n_jobs=-1)
            params = self.bay_opt(param_space)
            self.model.set_params(**params)
            return self
        elif self.model_name == 'xgboost':
            self.model = XGBRegressor(verbosity=1)
            params = self.bay_opt(param_space)
            self.model.set_params(**params)
            return self
        elif self.model_name == 'blr':
            self.model = BayesianRidge(verbose=1)
            params = self.bay_opt(param_space)
            self.model.set_params(**params)
            return self
        elif self.model_name == 'ak_struct_reg':
            self.model = ak.StructuredDataRegressor(
                max_trials=30,
                overwrite=True,
                metrics=['mse'],
                seed=42,
                tuner='bayesian',
                project_name='ak_struct_reg_trials')
            return self
        else:
            raise ValueError('Invalid model name')

    def fitcv(self, scale, pca=False, pca_n=10):
        kf = KFold(self.n_folds)
        performance = []

        for fold, (train_index, test_index) in enumerate(kf.split(self.X)):
            print(f'Now training fold {fold + 1}')
            X_train = self.X.loc[train_index, :]
            X_test = self.X.loc[test_index, :]
            y_train, y_test = self.y[train_index], self.y[test_index]
            if pca:
                print(f'Applying PCA of {pca_n} components')
                X_train = self.PCA_X(X_train, pca_n, train=True)
                X_test = self.PCA_X(X_test, pca_n)
                self.features = len(X_train.columns)
                print('PCA done')
            if scale:
                print('scaling data using StandardScaler')
                X_train = self.scale_X(data=X_train, train=True)
                X_test = self.scale_X(data=X_test)
                print('scaling done')
            # self.load_model()
            if self.model_name == 'ak_struct_reg':
                self.model.fit(
                    X_train.to_numpy(),
                    y_train.to_numpy(),
                    validation_split=0.2,
                    epochs=1000,
                    verbose=1)
                y_pred = self.model.predict(X_test.to_numpy()).flatten()
            else:
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

            # Compute and store the performance metrics for each fold
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            performance.append({
                'model': self.model_name,
                'n_folds': self.n_folds,
                'fold': fold + 1,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'r': np.corrcoef(y_pred, y_test)[0, 1],
                'y_pred': y_pred,
                'y_test': y_test})

        # Compute the average performance across all folds
        self.avg_performance = {
            'model': self.model_name,
            'n_folds': self.n_folds,
            'mae': sum(p['mae'] for p in performance) / len(performance),
            'mse': sum(p['mse'] for p in performance) / len(performance),
            'rmse': sum(p['rmse'] for p in performance) / len(performance),
            'r2': sum(p['r2'] for p in performance) / len(performance),
            'r': sum(p['r'] for p in performance) / len(performance),
        }
        return performance

    def get_avg_performance(self):
        return self.avg_performance

    def predict(self, X):
        return self.model.predict(X)

    def score(self, y, y_pred):
        return self.model.score(y, y_pred)

    def get_param_space(self):

        if self.model_name == 'ridge':
            param_space = {
                'alpha': np.arange(0.1, 20, 0.1),
                'tol': [1e-6],
                'fit_intercept': [True, False]
            }
        elif self.model_name == 'random_forest':
            param_space = {
                'n_estimators': np.arange(64, 1024, 16),
                'max_depth': [3, 5, 7, 9, 11],
                'max_samples': [0.3, 0.5, 0.8],
                'min_samples_split': np.linspace(.1, 1, 10, endpoint=True),
                'min_samples_leaf': np.linspace(.1, .5, 5, endpoint=True)
            }
        elif self.model_name == 'xgboost':
            param_space = {
                'eta': np.linspace(0.1, 0.4, 20, endpoint=True),
                'gamma': np.linspace(0, 20, 2, endpoint=True),
                'max_depth': np.arange(2, 12, 1),
                'min_child_weight': np.arange(0, 10, 1),
                'n_estimators': np.arange(32, 1024, 16),
                'tree_method': ['gpu_hist'],
                'objective': ['reg:squarederror']
            }
        elif self.model_name == "blr":
            param_space = {
                'n_iter': [1000],
                'alpha_1': [1e-4, 1e-5, 1e-6, 1e-7],
                'alpha_2': [1e-4, 1e-5, 1e-6, 1e-7],
                'lambda_1': [1e-4, 1e-5, 1e-6, 1e-7],
                'lambda_2': [1e-4, 1e-5, 1e-6, 1e-7]
            }
        else:
            raise ValueError(f"{self.model_name} is not a supported model")

        return param_space


def benchmark(
        X,
        y,
        scale,
        trans,
        model_name,
        pca=False,
        pca_n=3,
        scale_data=False,
        n_folds=5,
        embs_list=['emb']):
    pca_n = int(min(pca_n, min(len(X), len(X.columns))))
    regressor = RegressorCV(n_folds, X, y, embs_list=embs_list)
    start_time = time.time()
    regressor.set_model(model_name)
    performance = regressor.fitcv(
        scale=scale_data,
        pca=pca,
        pca_n=pca_n)
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_perf = regressor.get_avg_performance()
    avg_perf['elapsed_time'] = elapsed_time
    avg_perf['n_samples'] = len(X)
    avg_perf['n_features'] = len(regressor.X.columns)
    avg_perf['transformer'] = trans
    avg_perf['scale'] = scale
    avg_perf['X_pca'] = 'yes' if pca else 'no'
    avg_perf['X_pca_n'] = pca_n
    avg_perf['scaled_data'] = 'yes' if scale_data else 'no'
    del regressor
    return performance, avg_perf, elapsed_time


def print_time(
        exec_time,
        avg_performance):
    hours, rem = divmod(exec_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        (f'average performance: {avg_performance}\nexecution time:',
         f'{hours:0>2}:{minutes:0>2}:{seconds:05.2f}')
    )


def save_performance(performance,
                     avg_performance,
                     layer_str,
                     target,
                     delta_vec=False,
                     emb_PCA=False,
                     emb_PCA_n=3,
                     X_added_t1_score=False,
                     scale='noscale',
                     spec_folder=''):
    model_name = avg_performance['model']
    trans_name = avg_performance['transformer']
    features = avg_performance['n_features']
    pca = avg_performance['X_pca']
    pca_n = avg_performance['X_pca_n']
    scaled = avg_performance['scaled_data']
    delta_vec = 'yes' if delta_vec else 'no'
    emb_PCA = 'yes' if emb_PCA else 'no'
    X_added_t1_score = 'yes' if X_added_t1_score else 'no'
    file_end = (
        f'{target}_{trans_name}_{layer_str}_{model_name}_{features}'
        f'_pca_{pca}_{pca_n}_scaled_{scaled}_delta_vec_{delta_vec}'
        f'_X_added_t1_score_{X_added_t1_score}')
    df = pd.DataFrame(performance)
    spec_folder = f'{spec_folder}/'
    subfolder_path = (f'_results/performance/{spec_folder}{scale}/{target}/'
                      f'{trans_name}/{model_name}/')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    perf = f'{subfolder_path}performance_{file_end}'
    avg_perf = f'{subfolder_path}avg_performance_{file_end}'
    run_nbr = 0
    while any(os.path.exists(f'{path}_{run_nbr}.csv')
              for path
              in (perf, avg_perf)):
        run_nbr += 1
    df.to_csv(f'{perf}_{run_nbr}.csv',
              index=False)
    df = pd.DataFrame([avg_performance])
    df.to_csv(f'{avg_perf}_{run_nbr}.csv',
              index=False)
    summary_path = f'{subfolder_path}avg_performance_{target}.csv'
    df['run_nbr'] = run_nbr
    df['layers'] = layer_str
    df['target'] = target
    df['delta_vec'] = delta_vec
    df['emb_PCA'] = emb_PCA
    df['emb_pca_n'] = emb_PCA_n
    df['X_added_t1_score'] = X_added_t1_score
    if os.path.exists(summary_path):
        df_1 = pd.read_csv(summary_path)
        df = pd.concat([df_1, df]).reset_index(drop=True)
        df.to_csv(summary_path, index=False)
        df.to_html(summary_path.replace('.csv', '.html'))
    else:
        df.to_csv(summary_path, index=False)
        df.to_html(summary_path.replace('.csv', '.html'))
    return df
