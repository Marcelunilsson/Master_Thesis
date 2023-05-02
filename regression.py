# %%
import _lib.regression as reg
import pandas as pd
import time

# set parameters -------------------------------------------------------------
scales = ['hils', 'swls']

transformers = [
    'bert-base-uncased',
    'bert-large-uncased',
    'roberta-base',
    'roberta-large',
    # "xlnet-base-cased",
    # 'distilbert-base-uncased'
    ][1:]


# 'bert-base-uncased': 12 hidden layers
# 'bert-large-uncased': 24 hidden layers
# 'roberta-base': 12 hidden layers
# 'roberta-large': 24 hidden layers
# 'distilbert-base-uncased': 6 hidden layers
# "xlnet-base-cased": 12 hidden layers


model_names = [
    'ridge',
    'random_forest',
    # 'xgboost', to slow, removed from benchmark
    'blr',
    'ak_struct_reg']

scales = ['hils']
# transformers = ['bert-large-uncased']
model_names = ['ak_struct_reg']

# play with these parameters
target = ['delta', 'score'][1]

scale_data = True

# Only for Delta predictions
delta_vec = False
X_added_t1_score = True

# Load embeddings where each layer has gone through pca
pca_embs = False
pca_embs_n_list = [2, 16, 32, 64, 128, 256] if pca_embs else [2]

# PCA X
pca = True
pca_n = [2, 16, 32, 64, 128, 256][4]


# -----------------------------------------------------------------------------
def predict_save(
        scale,
        trans,
        model_name,
        target,
        scale_data,
        delta_vec,
        X_added_t1_score,
        pca_embs,
        pca_embs_n,
        pca,
        pca_n):
    t = ['t1', 't2']
    path = f'_data/embedding_dataframes/{scale}_{trans}/'
    layer_list = (
        [24, 25]
        if trans
        in ['bert-large-uncased', 'roberta-large']
        else [6, 7]
        if trans in ['distilbert-base-uncased']
        else [12, 13])
    layers = '_'.join([str(lay) for lay in layer_list])
    pca_suffix = f'_pca_{pca_embs_n}' if pca_embs else ''
    if target == 'delta':
        suffix = f'_delta_vec{pca_suffix}' if delta_vec else f'{pca_suffix}'
        embs_list = ([f'lay_{lay}' for lay in layer_list]
                     if delta_vec
                     else ([f't1_lay_{lay}' for lay in layer_list] +
                           [f't2_lay_{lay}' for lay in layer_list]))
        file_name = (
            f'{scale}_{target}{suffix}_layers'
            f'_{layers}_concat_embs_{trans}')
        df = pd.read_feather(
            f'{path}{file_name}.feather')
        df = df.sample(frac=1, random_state=42)
        y_col = f'delta_{scale}'
        y = df[y_col]
        X_cols = (
            ([f'{scale}_t1'] + list(df.columns[5:]))
            if X_added_t1_score
            else df.columns[5:])
        X = df[X_cols]
    else:
        file_name = (
            f'{scale}_{target}{pca_suffix}_{t[0]}'
            f'_layers_{layers}_concat_embs_{trans}')
        df_1 = pd.read_feather(
            f'{path}{file_name}.feather')
        file_name = (
            f'{scale}_{target}{pca_suffix}_{t[1]}'
            f'_layers_{layers}_concat_embs_{trans}')
        df_2 = pd.read_feather(
            f'{path}{file_name}.feather')
        cols = [f'{scale}', 'words'] + list(df_1.columns)[2:]
        df_1.columns = df_2.columns = cols
        embs_list = [f'lay_{lay}' for lay in layer_list]
        df = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
        df = df.sample(frac=1, random_state=42)
        y = df[scale]
        X = df[df.columns[2:]]
        target = 'score'

    performance, avg_performance, exec_time = reg.benchmark(
        X,
        y,
        scale,
        trans,
        model_name=model_name,
        pca=pca,
        pca_n=pca_n,
        scale_data=scale_data,
        embs_list=embs_list)
    avg_df = reg.save_performance(
        performance,
        avg_performance,
        layer_str=layers,
        target=target,
        delta_vec=delta_vec,
        emb_PCA=pca_embs,
        emb_PCA_n=pca_embs_n,
        X_added_t1_score=X_added_t1_score,
        scale=scale)
    reg.print_time(exec_time, avg_performance)
    return avg_df


t1 = time.time()
for scale in scales:
    for trans in transformers:
        for model_name in model_names:
            for pca_embs_n in pca_embs_n_list:
                avg_df = predict_save(
                    scale=scale,
                    trans=trans,
                    model_name=model_name,
                    target=target,
                    scale_data=scale_data,
                    delta_vec=delta_vec,
                    X_added_t1_score=X_added_t1_score,
                    pca_embs=pca_embs,
                    pca_embs_n=pca_embs_n,
                    pca=pca,
                    pca_n=pca_n)
t2 = time.time()
print('Done!')

print((
    f'Run with parameters:\n'
    f'scale_data: {scale_data}\n'
    f'delta_vec: {delta_vec}\n'
    f'X_added_t1_score: {X_added_t1_score}\n'
    f'pca_embs: {pca_embs}\n'
    f'pca_embs_n_list: {pca_embs_n_list}\n'
    f'scales: {scales}\n'
    f'transformers: {transformers}\n'
    f'model_names: {model_names}\n'))
seconds = (t2 - t1)
minutes, seconds = divmod(int(seconds), 60)
hours, minutes = divmod(minutes, 60)
print(f'Time: {hours} hours, {minutes} minutes, {seconds} seconds')

# %%
