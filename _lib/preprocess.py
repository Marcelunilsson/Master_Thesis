# %%
import os
import pickle as pkl

import pandas as pd
from _lib.metrics_stat import cohens_d
from _lib.metrics_stat import RCI
from _lib.metrics_stat import SID
from _lib.metrics_stat import ttest_p_val
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts


def simple_hils_swls(local_run=False):
    """Returns a curated data set

    Returns:
        DataSet: Curated data set
    """
    pre = '../' if local_run else ''
    df = pd.read_csv(f'{pre}_data/HILS_SWLS.csv')
    working_cols = ['_hilstotalt1',
                    '_hilstotalt2',
                    '_swlstotalt1',
                    '_swlstotalt2',
                    'harmony_t1',
                    'satisfaction_t1',
                    'harmony_t2',
                    'satisfaction_t2']
    return df[working_cols]


def split_data(df, X_name, y_name, test_size=.33):
    """Just performs sklearns train test split on the data frame

    Args:
        df (DataSet): a pandas dataset
        X_name (str): feature column name
        y_name (str): Label column name
        test_size (float): test fraction of dataset. Defaults to .33.

    Returns:
        DataSets: X_train, X_test, y_train, y_test
    """
    return tts(
        df[X_name],
        df[y_name],
        test_size=test_size,
        random_state=101
    )


def data_set_two():
    """Load a data set I have not looked that much at yet

    Returns:
        DataSet: The data_set_2.csv
    """
    df = pd.read_csv('_data/data_set_2.csv',
                     index_col=0)
    return df


def one_construct(construct):
    """Generates a simple data set from one construct.
    the data set has the columns ['t1', 't2', 'deltas', 'words_t1', 'words_t2']

    Args:
        construct (str): What construct ['hils', 'swls']

    Returns:
        DataSet: The simple dataset
    """
    data_frame = simple_hils_swls()
    words = 'harmony_' if construct == 'hils' else 'satisfaction_'
    construct_df = pd.DataFrame()
    construct_df['t1'] = data_frame[f'_{construct}totalt1']
    construct_df['t2'] = data_frame[f'_{construct}totalt2']
    construct_df['deltas'] = add_deltas(data_frame=data_frame,
                                        construct=construct)
    construct_df['words_t1'] = data_frame[f'{words}t1']
    construct_df['words_t2'] = data_frame[f'{words}t2']
    return construct_df


def time_invariant_oc(construct):
    """Generates a simple data set from one construct.
    the data set has the columns ['words', construct]

    Args:
        construct (str): Scale to use ['hils', 'swls']

    Returns:
        _type_: The simple dataset
    """
    df = pd.read_csv('_data/HILS_SWLS.csv')
    words = 'harmony_' if construct == 'hils' else 'satisfaction_'
    t1_df, t2_df = pd.DataFrame(), pd.DataFrame()
    t1_df['words'] = df[f'{words}t1'].apply(lambda x: x.lower())
    t1_df[construct] = df[f'_{construct}totalt1']
    t2_df['words'] = df[f'{words}t2'].apply(lambda x: x.lower())
    t2_df[construct] = df[f'_{construct}totalt2']
    return pd.concat([t1_df, t2_df], ignore_index=True)


def string_set(df):
    """Generates a string and a set from a data set

    Args:
        df (DataSet): The data set

    Returns:
        str, set: The string and set
    """
    string = ' '.join(df['words'].tolist())
    u_words = set(string.split())
    return string, u_words


def word_val(construct):
    """Generates a simple data set from one construct, showing each word in the
    data set with its mean construct value, with its count.
    the data set has the columns ['word', 'mean_construct', 'word_count']

    Args:
        construct (str): Scale to use ['hils', 'swls']

    Returns:
        DataFrame: The simple data set
    """
    df = time_invariant_oc(construct)
    string, u_words = string_set(df)
    df['word_list'] = df['words'].apply(lambda x: x.lower().split())
    df_word_val = pd.DataFrame()
    df_word_val['word'] = list(set)
    df_word_val[f'mean_{construct}'] = (
        df_word_val['word'].apply(
            lambda x:
                df[df['word_list'].apply(
                    lambda y: x.lower() in y)][construct].mean()
        )
    )
    df_word_val['word_count'] = df_word_val['word'].apply(
        lambda x: string.count(x))
    return df_word_val


def hils_swls_concat():
    """Concatenates hils and swls and add deltas and RCI

    Returns:
        DataFrame: the concatenated data set
    """
    df_1 = one_construct(construct='hils')
    df_1['RCI'], df_1['rel'], _, _ = RCI(
        df_1,
        col_1='t1',
        col_2='t2'
    )
    df_1['construct'] = ['hils'] * len(df_1)
    df_2 = one_construct(construct='swls')
    df_2['RCI'], df_2['rel'], _, _ = RCI(
        df_2,
        col_1='t1',
        col_2='t2'
    )
    df_2['construct'] = ['swls'] * len(df_2)
    return pd.concat([df_1, df_2],
                     ignore_index=True)


def hils_swls_metrics():
    """Simple hils and swls with the metrics (RCI, SID, deltas)
    and a dictionary with the stats (cohens d, p-value)

    Returns:
        (DataFrame, dict): the Dataframe with metrics and dict with stats
    """
    df = simple_hils_swls()
    scales = ['hils', 'swls']
    stats = {}
    for scale in scales:
        t1, t2 = f'_{scale}totalt1', f'_{scale}totalt2'
        (df[f'RCI_{scale}'],
         df[f'rel_{scale}'],
         stats[f'CI_{scale}'],
         stats[f'r_{scale}']) = RCI(
            df,
            t1,
            t2
        )
        df[f'SID_{scale}'] = SID(
            df,
            t1,
            t2
        )
        stats[f'd_{scale}'] = cohens_d(df,
                                       t1,
                                       t2
                                       )
        stats[f'p_{scale}'] = ttest_p_val(df,
                                          t1,
                                          t2)
        df[f'delta_{scale}'] = add_deltas(df, scale)
    return df, stats


def add_deltas(data_frame, construct):
    return (data_frame[f'_{construct}totalt2']
            - data_frame[f'_{construct}totalt1'])


def big_delta_dataset(construct):
    df = one_construct(construct)
    df = df.drop(['deltas'], axis=1)
    df.columns = [
        f'{construct}_t1',
        f'{construct}_t2',
        'words_t1',
        'words_t2']
    n = len(df)
    big_delta_df = pd.DataFrame(columns=df.columns)
    for i, row in df.iterrows():
        df1 = pd.DataFrame({
            f'{construct}_t1': [row[f'{construct}_t1']] * n,
            f'{construct}_t2': df[f'{construct}_t2'].tolist(),
            'words_t1': [row['words_t1']] * n,
            'words_t2': df['words_t2'].tolist()})
        df2 = pd.DataFrame({
            f'{construct}_t1': [row[f'{construct}_t2']] * n,
            f'{construct}_t2': df[f'{construct}_t1'].tolist(),
            'words_t1': [row['words_t2']] * n,
            'words_t2': df['words_t1'].tolist()})
        big_delta_dataset = pd.concat([big_delta_df, df1, df2],
                                      ignore_index=True)
    big_delta_dataset = big_delta_dataset.drop_duplicates()
    big_delta_dataset[f'delta_{construct}'] = (
            big_delta_dataset[f'{construct}_t2'] -
            big_delta_dataset[f'{construct}_t1']
        )
    big_delta_dataset.to_feather(
        f'_data/{construct}_delta_dataset.feather')
    return big_delta_dataset


class emb_df:
    def __init__(self):
        self.set_construct()
        self.set_model()
        self.set_path()
        self.set_subfolder()

    def set_model(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        return self

    def set_construct(self, construct='hils'):
        self.construct = construct
        return self

    def set_path(self, path='_data/'):
        self.path = path
        return self

    def set_subfolder(self,
                      subfolder=''):
        if subfolder == '':
            self.subfolder = ('embedding_dataframes/'
                              f'{self.construct}_{self.model_name}/')
        else:
            self.subfolder = subfolder
        return self

    def load_dataset(self,
                     time='t1',
                     delta=False,
                     file_name='simple_hils_swls_metrics'):
        df = pd.read_feather(f'{self.path}{file_name}.feather')
        cols = (
            [f'hils_{time}', f'harmony_{time}']
            if self.construct == 'hils'
            else [f'swls_{time}', f'satisfaction_{time}']
        ) if not delta else (
            ['hils_t1',
             'harmony_t1',
             'hils_t2',
             'harmony_t2',
             'delta_hils']
            if self.construct == 'hils'
            else [
                'swls_t1',
                'satisfaction_t1',
                'swls_t2',
                'satisfaction_t2',
                'delta_swls']
        )
        return df[cols], cols

    def load_embs(self, file_name, sub_path=''):
        file_path = f'{self.path}{sub_path}{file_name}.pkl'
        with open(file_path, 'rb') as f:
            return pkl.load(f)

    def emb_list_cols(
            self,
            df,
            word_col,
            embs,
            time,
            layer,
            pca=False,
            pca_n=3):
        emb_list = [
                embs[word][f'layer_{layer}']
                for word
                in df[word_col]]
        if pca:
            pca_path = f'{self.path}pca_models/'
            pca_file = (f'{self.construct}_{self.model_name}'
                        f'_pca_{pca_n}_layer_{layer}.pkl')
            if os.path.exists(f'{pca_path}{pca_file}'):
                with open(f'{pca_path}{pca_file}', 'rb') as f:
                    pca = pkl.load(f)
                emb_list = pca.transform(emb_list)
            else:
                pca = PCA(n_components=pca_n)
                emb_list = pca.fit_transform(emb_list)
                if not os.path.exists(pca_path):
                    os.makedirs(pca_path)
                with open(f'{pca_path}{pca_file}', 'wb') as f:
                    pkl.dump(pca, f)
        emb_cols = [
            f'emb_{i}_{time}_lay_{layer}'
            for i
            in range(len(emb_list[0]))]
        return emb_list, emb_cols

    def add_embs(self,
                 df,
                 word_col,
                 embs,
                 layers,
                 time,
                 pca=False,
                 pca_n=3):
        for layer in layers:
            emb_list, emb_cols = self.emb_list_cols(
                df=df,
                word_col=word_col,
                embs=embs,
                time=time,
                layer=layer,
                pca=pca,
                pca_n=pca_n)
            emb_df = pd.DataFrame(emb_list, columns=emb_cols)
            df = pd.concat([df, emb_df], axis=1)
        return df

    def save_embs_df(self, df, layers, delta=True, suffix=''):
        data_type = 'delta' if delta else 'score'
        layer_str = '_'.join([str(layer) for layer in layers])
        save_path = f'{self.path}{self.subfolder}'
        file_name = (
            f'{self.construct}_{data_type}_{suffix}'
            f'layers_{layer_str}_concat_embs_{self.model_name}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_feather(f'{save_path}{file_name}.feather')

    def load_score_608_df(self):
        df = pd.read_feather(
            f'{self.path}score_608.feather')
        cols = (['hils', 'harmony'] if self.construct == 'hils'
                else ['swls', 'satisfaction'])
        return df[cols], cols

    def score_embs_concat_608(
            self,
            layers,
            save=True,
            pca=False,
            pca_n=3):
        embs = self.load_embs(
            (f'score_608_{self.construct}_t1_{self.model_name}'
             f'_embs_dict'),
            sub_path='embs_dicts/final_model_test/')
        df, cols = self.load_score_608_df()
        df = self.add_embs(
            df=df,
            word_col=cols[1],
            embs=embs,
            layers=layers,
            time='t1',
            pca=pca,
            pca_n=pca_n)
        if save:
            layer_str = '_'.join([str(layer) for layer in layers])
            save_path = f'{self.path}{self.subfolder}'
            pca_str = f'_pca_{pca_n}' if pca else ''
            file_name = (
                f'{self.construct}_score_608_t1_layers_{layer_str}'
                f'_concat_embs_{self.model_name}{pca_str}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_feather(f'{save_path}{file_name}.feather')
        return df

    def score_embs_concat(self,
                          layers,
                          time,
                          save=True,
                          pca=False,
                          pca_n=3,
                          emb_sub_path=''):
        embs = self.load_embs(
            (f'hils_swls_delta_{self.construct}_{time}'
             f'_{self.model_name}_embs_dict'),
            sub_path=f'embs_dicts/{emb_sub_path}'
            )
        df, cols = self.load_dataset(time)
        df = self.add_embs(
            df=df,
            word_col=cols[1],
            embs=embs,
            layers=layers,
            time=time,
            pca=pca,
            pca_n=pca_n)
        suffix = f'pca_{pca_n}_{time}_' if pca else f'{time}_'
        if save:
            self.save_embs_df(df, layers, delta=False, suffix=suffix)
        return df

    def load_t1_t2_embs(
            self):
        embs_t1 = self.load_embs(
            f'hils_swls_delta_{self.construct}_t1_{self.model_name}_embs_dict',
            sub_path='embs_dicts/')
        embs_t2 = self.load_embs(
            f'hils_swls_delta_{self.construct}_t2_{self.model_name}_embs_dict',
            sub_path='embs_dicts/')
        return embs_t1, embs_t2

    def delta_embs_concat(
            self,
            layers,
            save=True,
            pca=False,
            pca_n=3):
        embs_t1, embs_t2 = self.load_t1_t2_embs()
        df, cols = self.load_dataset(delta=True)
        df = self.add_embs(
            df=df,
            word_col=cols[1],
            embs=embs_t1,
            layers=layers,
            time='t1',
            pca=pca,
            pca_n=pca_n)
        df = self.add_embs(
            df=df,
            word_col=cols[3],
            embs=embs_t2,
            layers=layers,
            time='t2',
            pca=pca,
            pca_n=pca_n)
        suffix = f'pca_{pca_n}_' if pca else ''
        if save:
            self.save_embs_df(df, layers, suffix=suffix)
        return df

    def delta_vector_embs(
            self,
            layers,
            save=True,
            pca=False,
            pca_n=3):
        embs_t1, embs_t2 = self.load_t1_t2_embs()
        df, cols = self.load_dataset(delta=True)
        for layer in layers:
            emb_list_t1, _ = self.emb_list_cols(
                df=df,
                word_col=cols[1],
                embs=embs_t1,
                time='t1',
                layer=layer,
                pca=pca,
                pca_n=pca_n)
            emb_list_t2, _ = self.emb_list_cols(
                df=df,
                word_col=cols[3],
                embs=embs_t2,
                time='t2',
                layer=layer,
                pca=pca,
                pca_n=pca_n)
            delta_vec = [[x-y for x, y in zip(t2_emb, t1_emb)]
                         for t2_emb, t1_emb
                         in zip(emb_list_t2, emb_list_t1)]
            delta_vec_cols = [f'delta_vec_{i}_lay_{layer}'
                              for i
                              in range(len(delta_vec[0]))]
            delta_vec_df = pd.DataFrame(delta_vec, columns=delta_vec_cols)
            df = pd.concat([df, delta_vec_df], axis=1)
            suffix = f'delta_vec_pca_{pca_n}_' if pca else 'delta_vec_'
            if save:
                self.save_embs_df(df, layers, delta=True, suffix=suffix)
            else:
                return df


def get_layer_str(transformer):
    layer_list = (
        [24, 25]
        if transformer
        in ['bert-large-uncased', 'roberta-large']
        else [6, 7]
        if transformer in ['distilbert-base-uncased']
        else [12, 13])
    layer_str = '_'.join([str(i) for i in layer_list])
    return layer_str, layer_list


def get_608_embs(
        scale,
        transformer,
        pca,
        pca_n):
    pca_str = f'_pca_{pca_n}' if pca else ''
    emb_path = '_data/embedding_dataframes/'
    sub_path = f'{scale}_{transformer}/'
    layer_str, lay_list = get_layer_str(transformer)
    score_608_file = (
        f'score_608/{sub_path}{scale}_score_608_t1_layers_{layer_str}'
        f'_concat_embs_{transformer}{pca_str}.feather')
    score_608_embs = pd.read_feather(
        f'{emb_path}{score_608_file}').sample(frac=1)
    score_608_embs.columns = [
        s.replace('_t1', '')
        for s
        in score_608_embs.columns]
    return score_608_embs, lay_list


def Xy_608(scale, transformer, pca, pca_n):
    hs_608_embs, lay_list = get_608_embs(
        scale,
        transformer,
        pca,
        pca_n)
    words = 'harmony' if scale == 'hils' else 'satisfaction'
    X = hs_608_embs.drop(columns=[f'{scale}', f'{words}'])
    y = hs_608_embs[f'{scale}'].astype('float')
    lay_list = [f'lay_{i}' for i in lay_list]
    return X, y, lay_list


def get_476_embs(
        scale,
        transformer,
        pca,
        pca_n,
        t,
        target):
    pca_str = f'_pca_{pca_n}' if pca else ''
    emb_path = '_data/embedding_dataframes/'
    sub_path = f'{scale}_{transformer}/'
    layer_str, lay_list = get_layer_str(transformer)
    t_str = 't1' if t == 1 else 't2'
    hs_476_file = (
        f'final_model_test/{sub_path}{scale}_{target}'
        f'{pca_str}_{t_str}_layers_{layer_str}_'
        f'concat_embs_{transformer}.feather'
        )
    hs_476_embs = pd.read_feather(
        f'{emb_path}{hs_476_file}').sample(frac=1)
    hs_476_embs.columns = [
        s.replace('_{t_str}', '')
        for s
        in hs_476_embs.columns]
    lay_list = [f'lay_{i}' for i in lay_list]
    return hs_476_embs, lay_list


def Xy_476(
        scale,
        transformer,
        pca,
        pca_n,
        target,
        t,
        concat=False):
    hs_embs_list = [
        get_476_embs(
            scale,
            transformer,
            pca,
            pca_n,
            time,
            target)
        for time
        in [1, 2]]
    hs_476_embs = concat_rows(hs_embs_list) if concat else hs_embs_list[t-1]
    words = 'harmony' if scale == 'hils' else 'satisfaction'
    X = hs_476_embs.drop(columns=[f'{scale}', f'{words}'])
    y = hs_476_embs[f'{scale}'].astype('float')
    return X, y


def concat_rows(df_list):
    return pd.concat(df_list, axis=0).reset_index(drop=True).sample(frac=1)
