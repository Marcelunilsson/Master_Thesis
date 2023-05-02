import numpy as np
import tensorflow as tf
from _lib import preprocess as pp
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import TFAutoModel
from transformers import TFRobertaModel


# from transformers import RobertaTokenizer, TFRobertaModel
# from _lib import preprocess as pp
# from transformers import RobertaTokenizerFast
def map_func(input_ids, masks, labels):
    # we convert our three-item tuple into a two-item tuple where the
    # input item is a dictionary
    return map_X(input_ids, masks), labels


def map_X(input_ids, masks):
    return {'input_ids': input_ids, 'attention_mask': masks}


def t1_t2(construct, train_frac=.9):
    df = pp.delta_dataset(construct=construct)
    train = df.sample(frac=train_frac)
    test = df.drop(train.index)
    y_name = f'_{construct}totalt1'
    X_name = 'harmony_t1' if construct == 'hils' else 'satisfaction_t1'
    return train, test, X_name, y_name


def simple_delta(construct, train_frac=.9):
    df = pp.simple_delta(construct=construct)
    train = df.sample(frac=train_frac)
    test = df.drop(train.index)
    y_name = 'Delta'
    X_name = 'words'
    return train, test, X_name, y_name


def time_inv(construct, train_frac=.9):
    df = pp.time_invariant_oc(construct=construct)
    train = df.sample(frac=train_frac)
    test = df.drop(train.index)
    x_name = 'words'
    y_name = construct
    return train, test, x_name, y_name


def prep_X_roberta(data, X_name, SEQ_LEN):
    model = 'roberta-large'
    toke = RobertaTokenizer.from_pretrained(model)
    NUM_SAMPLES = len(data)
    Xids = np.zeros((NUM_SAMPLES, SEQ_LEN))
    Xmask = np.zeros((NUM_SAMPLES, SEQ_LEN))
    for i, words in enumerate(data[X_name]):
        tokens = toke.encode_plus(
            words,
            max_length=SEQ_LEN,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        Xids[i, :] = tokens['input_ids']
        Xmask[i, :] = tokens['attention_mask']
    return Xids, Xmask


def roberta_ds(
    predict='val',
    construct='hils',
    batch_size=8,
    train_val_split=.7,
    SEQ_LEN=16
):
    if predict == 'val':
        print(f'Predicting {construct} value')
        train, test, X_name, y_name = time_inv(construct)
    elif predict == 'change':
        print(f'Predicting {construct} deltas')
        train, test, X_name, y_name = simple_delta(construct=construct)
    print('Building Xids and Xmask')
    Xids, Xmask = prep_X_roberta(train, X_name, SEQ_LEN)
    testX_ids, testX_mask = prep_X_roberta(test, X_name, SEQ_LEN)
    test_X = np.array(
        [map_X(np.reshape(X_ids, (1, SEQ_LEN)),
               np.reshape(X_mask, (1, SEQ_LEN)))
         for X_ids, X_mask
         in zip(testX_ids, testX_mask)]
    )
    test_y = test[y_name].values
    labels = train[y_name].values
    print('Casting to tensorflow data set')
    train_ds = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
    train_ds = train_ds.map(map_func)
    train_ds = train_ds.shuffle(1000).batch(
        batch_size,
        drop_remainder=True
    )
    print('Splitting to validation, train and test data sets')
    size = int((len(train)/batch_size)*train_val_split)
    val_ds = train_ds.skip(size)
    return train_ds.take(size), val_ds, test_X, test_y


def get_optimizer(optimizer='adam'):
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            decay=1e-6
        )
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(
            learning_rate=1e-4,
            momentum=1e-6
        )
    return opt


def inputs(name, SEQ_LEN, dtype='int32'):
    return tf.keras.layers.Input(
        shape=(SEQ_LEN,),
        name=name,
        dtype='int32'
    )


def roberta_model(SEQ_LEN=16, optimizer='sgd'):
    roberta = TFRobertaModel.from_pretrained('roberta-large')
    input_ids = inputs(name='input_ids', SEQ_LEN=SEQ_LEN)
    mask = inputs(name='attention_mask', SEQ_LEN=SEQ_LEN)
    embeddings = roberta.roberta(
        input_ids,
        attention_mask=mask)[1]
    bn_1 = tf.keras.layers.BatchNormalization()(embeddings)
    x_1 = tf.keras.layers.Dense(
        2048,
        activation='relu',
        name='Dense_1')(bn_1)
    # bn_2 = tf.keras.layers.BatchNormalization()(x_1)
    # x_2 = tf.keras.layers.Dense(
    #     1024,
    #     activation='relu',
    #     name='Dense_2')(bn_2)
    # bn_3 = tf.keras.layers.BatchNormalization()(x_2)
    y = tf.keras.layers.Dense(1, name='outputs')(x_1)
    model = tf.keras.Model(
        inputs=[input_ids, mask],
        outputs=y
    )
    model.layers[2].trainable = False
    opt = get_optimizer(optimizer)
    loss = tf.keras.losses.MeanSquaredError(
            reduction='auto',
            name='mean_squared_error'
        )
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model


def X_tokens_bert(X):
    seq_len = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokens = tokenizer(
        X.tolist(),
        max_length=seq_len,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='np'
    )
    return tokens


def make_ds(X_tokens, y, batch_size=8):
    ds = tf.data.Dataset.from_tensor_slices(
        (X_tokens['input_ids'], X_tokens['attention_mask'], y)
    )
    ds = ds.map(map_func=map_func)
    ds = ds.shuffle(10000).batch(
        batch_size=batch_size,
        drop_remainder=True
    )
    return ds


def train_val(ds, tokens, batch_size=8, split=.9):
    size = int((tokens['input_ids'].shape[0] / batch_size) * split)
    return ds.take(size), ds.skip(size)


def bert_model_1():
    bert = TFAutoModel.from_pretrained('bert-base-cased')
    input_ids = tf.keras.layers.Input(shape=(512, ),
                                      name='input_ids',
                                      dtype='int32')
    mask = tf.keras.layers.Input(shape=(512, ),
                                 name='attention_mask',
                                 dtype='int32')
    embeddings = bert.bert(input_ids, attention_mask=mask)[1]
    x = tf.keras.layers.Dense(1024,
                              activation='relu')(embeddings)
    y = tf.keras.layers.Dense(1,
                              name='outputs')(x)
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    model.layers[2].trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5,
                                         decay=1e-6)
    loss = tf.keras.losses.MeanSquaredError(
        reduction='auto',
        name='mean_squared_error'
    )
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


def bert_regression(X_train,
                    y_train,
                    batch_size=8):
    tokens = X_tokens_bert(X_train)
    ds = make_ds(tokens, y_train, batch_size)
    train_ds, val_ds = train_val(ds, tokens)
    model = bert_model_1()
    return model, train_ds, val_ds
