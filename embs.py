# %%
import _lib.preprocess as pp


def hils_swls_set(
        model,
        pca,
        pca_n,
        construct,
        time):
    ed = pp.emb_df().set_construct(construct).set_model(model)
    subfolder = ('embedding_dataframes/final_model_test/'
                 f'{construct}_{model}/')
    ed.set_subfolder(subfolder)
    layer_list = (
        [24, 25]
        if model
        in ['bert-large-uncased', 'roberta-large']
        else [6, 7]
        if model in ['distilbert-base-uncased']
        else [12, 13])
    for t in time:
        df = ed.score_embs_concat(
            layers=layer_list,
            time=t,
            save=True,
            pca=pca,
            pca_n=pca_n)
    df_delta = ed.delta_embs_concat(
        layers=layer_list,
        save=True,
        pca=pca,
        pca_n=pca_n)
    df_delta_vec = ed.delta_vector_embs(
        layers=layer_list,
        save=True,
        pca=pca,
        pca_n=pca_n)
    return df, df_delta, df_delta_vec


def score_608_set(
        model,
        pca,
        pca_n,
        construct):
    ed = pp.emb_df().set_construct(construct).set_model(model)
    subfolder = ('embedding_dataframes/score_608/'
                 f'{construct}_{model}/')
    ed.set_subfolder(subfolder)
    layer_list = (
        [24, 25]
        if model
        in ['bert-large-uncased', 'roberta-large']
        else [6, 7]
        if model in ['distilbert-base-uncased']
        else [12, 13])
    df = ed.score_embs_concat_608(
        layers=layer_list,
        save=True,
        pca=pca,
        pca_n=pca_n)
    return df


# %%
score_608 = False
time = ['t1', 't2']
# time = ['t1']
constructs = ['hils', 'swls']
models = [
    'bert-base-uncased',
    'bert-large-uncased',
    'roberta-base',
    'roberta-large',
    'distilbert-base-uncased',
    "xlnet-base-cased"]
# 'bert-base-uncased': 12 hidden layers
# 'bert-large-uncased': 24 hidden layers
# 'roberta-base': 12 hidden layers
# 'roberta-large': 24 hidden layers
# 'distilbert-base-uncased': 6 hidden layers
# "xlnet-base-cased": 12 hidden layers
pca = False
pca_ns = [2, 16, 32, 64, 128, 256] if pca else [2]
for construct in constructs:
    for model in models:
        for pca_n in pca_ns:
            if score_608:
                df = score_608_set(
                    model=model,
                    pca=pca,
                    pca_n=pca_n,
                    construct=construct)
            else:
                df, df_delta, df_delta_vec = hils_swls_set(
                    model=model,
                    pca=pca,
                    pca_n=pca_n,
                    construct=construct,
                    time=time)
