# %%
import os
import pickle

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.utils import logging


class torch_embs:
    """
        torch_embs: A class that computes contextualized
        word embeddings using transformers.

        Attributes:
            model_name (str): The name of the transformer model to use.
            config (AutoConfig): The configuration object for the model.
            tokenizer (AutoTokenizer): The tokenizer object for the model.
            model (AutoModel): The model object.
            device_name (str): The name of the device to use for computation.
            device_num (int): The number of the device to use for computation.
            logging_level (str): The logging level to use.

            A brief list of useable models can be found here:
                model_list = [
                    "bert-base-uncased",
                    "bert-base-cased",
                    "bert-large-uncased",
                    "bert-large-cased",
                    "roberta-base",
                    "roberta-large",
                    "xlnet-base-cased",
                    "xlnet-large-cased",
                    "albert-base-v2",
                    "albert-large-v2",
                    "distilbert-base-uncased",
                    "distilbert-base-cased",
                ]

            A list of device names can be found here:
                device_list = [
                    "cpu",
                    "gpu",
                    "cuda",
                    "gpu:0",
                    "cuda:0",
                    "gpu:1",
                    "cuda:1",
                    # and so on for other GPU/CUDA devices
                ]

            A list of logging levels can be found here:
                logging_level_list = [
                    "critical": "Only critical messages will be logged.",
                    "error": "Errors and critical messages will be logged.",
                    "warning": "Warnings, errors, and critical messages will
                        be logged.",
                    "info": "Information, warnings, errors, and critical
                        messages will be logged.",
                    "debug": "Debug information, information, warnings,
                        errors, and critical messages will be logged.",
                ]


        Methods:
            set_model(model_name: str, device_name: str):
                Initializes the model, tokenizer, and configuration
                objects with the specified model name, and moves the
                model to the specified device.
            set_device(device_name: str):
                Moves the model to the specified device.
            set_logging_level(logging_level: str):
                Sets the logging level to the specified value.
            get_embs(text_strings: List[str],
                     layers: str,
                     max_len: int,
                     chunk_size: int):
                Computes contextualized word embeddings for the given
                list of text strings.
            get_sent_embs(text_strings: List[str],
                          layers: str,
                          max_len: int,
                          chunk_size: int):
                Computes contextualized sentence embeddings for the
                given list of text strings.
    """

    def __init__(self, model_name: str,
                 device_name: str = 'cpu',
                 logging_level: str = 'warning'):
        """
            Initialize a `torch_embs` instance.

            Parameters
            ----------
            model_name : str
                The name of the transformer model to use. This should
                be one of the models listed in the Transformers documentation:
                https://huggingface.co/transformers/pretrained_models.html
            device_name : str, optional
                The name of the device to use for computation. This can be
                'cpu', 'gpu', 'cuda', 'gpu:k', or 'cuda:k',
                where k is an integer
                specifying the device
                number. The default is 'cpu'.
            logging_level : str, optional
                The logging level to use. This can be 'critical',
                'error', 'warning', 'info', or 'debug'.
                The default is 'warning'.
        """
        self.model_name = model_name
        self.set_model(model_name, device_name)
        self.set_logging_level(logging_level)

    def set_model(self,
                  model_name: str,
                  device_name: str = 'cpu'):
        # set the model name
        self.model_name = model_name
        # load the model configuration and tokenizer using the AutoConfig
        # and AutoTokenizer classes from the transformers library
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # load the model using the AutoModel class from
        # the transformers library
        # and the previously loaded configuration
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=self.config)
        # set the device for computation using the set_device() method
        self.set_device(device_name)
        # return the torch_embs instance itself to allow for method chaining
        return self

    def set_device(self,
                   device_name: str):
        self.device_name = device_name.lower()
        # Check if device_name does not start with 'cpu', 'gpu', or 'cuda'
        logic = (not self.device_name.startswith('cpu')
                 and not self.device_name.startswith('gpu')
                 and not self.device_name.startswith('cuda'))
        if logic:
            print(
                "device must be 'cpu', 'gpu', 'cuda',"
                "or of the form 'gpu:k' or 'cuda:k'")
            print("\twhere k is an integer value for the device")
            print("Trying CPUs")
            self.device_name = 'cpu'
        self.device_num = -1
        if self.device_name != 'cpu':
            if torch.cuda.is_available():
                if self.device_name == 'gpu' or self.device_name == 'cuda':
                    self.device_name = 'cuda'
                    self.device_num = list(range(torch.cuda.device_count()))[0]
                else:
                    # Try to parse device number from device_name
                    try:
                        self.device_num = int(device_name.split(":")[-1])
                        self.device_name = 'cuda:' + str(self.device_num)
                    except ValueError:
                        print(
                            "Invalid device number, "
                            "using first available GPU")
                        self.device_name = 'cuda'
                        self.device_num = list(
                            range(torch.cuda.device_count()))[0]
            else:
                print("CUDA (GPU) is not available, using CPU")
                self.device_name = "cpu"
                self.device_num = -1
            self.model.to(self.device_name)
        else:
            # use CPU
            self.device_name = "cpu"
            self.device_num = -1
        return self

    def set_logging_level(self,
                          logging_level: str):
        self.logging_level = logging_level.lower()
        # default level is warning, which is in between "error" and "info"
        if self.logging_level in ['warn', 'warning']:
            logging.set_verbosity_warning()
        elif self.logging_level == "critical":
            logging.set_verbosity(50)
        elif self.logging_level == "error":
            logging.set_verbosity_error()
        elif self.logging_level == "info":
            logging.set_verbosity_info()
        elif self.logging_level == "debug":
            logging.set_verbosity_debug()
        else:
            print(
                f"Warning: Logging level {self.logging_level}"
                "is not an option.")
            print("\tUse one of: critical, error, warning, info, debug")
        return self

    def get_embs(
            self,
            text_strings,
            layers='all',
            return_tokens=True,
            max_token_to_sentence=4,
            tokenizer_parallelism=True,
            model_max_length=None):
        if tokenizer_parallelism:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not isinstance(text_strings, list):
            text_strings = [text_strings]

        if layers != 'all':
            if not isinstance(layers, list):
                layers = [layers]
            layers = [int(i) for i in layers]

        all_embs = []
        all_toks = []

        for text_string in text_strings:
            # if length of text_string is > max_token_to_sentence*4
            # embedd each sentence separately
            if len(text_string) > max_token_to_sentence*4:
                sentence_batch = [s for s in sent_tokenize(text_string)]
                if model_max_length is None:
                    batch = self.tokenizer(
                        sentence_batch,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True)
                else:
                    batch = self.tokenizer(
                        sentence_batch,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=model_max_length)
                input_ids = torch.tensor(batch["input_ids"])
                attention_mask = torch.tensor(batch['attention_mask'])
                if self.device_name != 'cpu':
                    input_ids = input_ids.to(self.device_name)
                    attention_mask = attention_mask.to(self.device_name)

                if return_tokens:
                    tokens = []
                    for ids in input_ids:
                        tokens.extend(
                            [
                                token
                                for token
                                in self.tokenizer.convert_ids_to_tokens(ids)
                                if token != '[PAD]'
                                ]
                            )
                    all_toks.append(tokens)

                with torch.no_grad():
                    hidden_states = self.model(
                        input_ids,
                        attention_mask=attention_mask)[-1]
                    if layers != 'all':
                        hidden_states = [
                            hidden_states[layer]
                            for layer
                            in layers]
                    hidden_states = [h.tolist() for h in hidden_states]

                sent_embedding = []
                # iterate over layers
                for layer in range(len(hidden_states)):
                    layer_embedding = []
                    # iterate over sentences
                    for sentence in range(len(hidden_states[layer])):
                        sent_list = [
                            tok
                            for ii, tok
                            in enumerate(hidden_states[layer][sentence])
                            if attention_mask[sentence][ii] > 0]
                        layer_embedding.extend(sent_list)
                    sent_embedding.append(layer_embedding)

                all_embs.append([[layer] for layer in sent_embedding])
            else:
                input_ids = self.tokenizer.encode(
                    text_string,
                    add_special_tokens=True)
                if return_tokens:
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

                if self.device_name != 'cpu':
                    input_ids = torch.tensor([input_ids]).to(self.device_name)
                else:
                    input_ids = torch.tensor([input_ids])

                with torch.no_grad():
                    hidden_states = self.model(input_ids)[-1]
                    if layers != 'all':
                        hidden_states = [
                            hidden_states[layer]
                            for layer
                            in layers]
                    hidden_states = [
                        h.tolist()
                        for h
                        in hidden_states]
                    all_embs.append(hidden_states)
                    if return_tokens:
                        all_toks.append(tokens)

        if return_tokens:
            return all_embs, all_toks
        else:
            return all_embs

    def get_embs_dict(self, embs, df, word_col):
        embs_dict = {}
        # iterate over word index in df and embs
        for i in range(len(embs)):
            layers_dict = {
                f'layer_{layer+1}': self.avg_embs(embs[i][layer][0])
                for layer
                in range(len(embs[i]))}
            embs_dict[df[word_col][i]] = layers_dict
        return embs_dict

    def avg_embs(self, layer):
        return [sum(col)/len(col) for col in zip(*layer[1:-1])]

    def save_embs_dict_to_pickle(
            self,
            embs_dict,
            file_name,
            save_path='../_data/'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_string = (
            f'{save_path}{file_name}.pkl'
        )
        with open(file_string, 'wb') as f:
            pickle.dump(embs_dict, f)
        print(f'Embeddings saved to {file_string}')

    def load_embs_dict_from_pickle(
            self,
            file_name,
            path='../_data/'):
        with open(f'{path}{file_name}.pkl', 'rb') as f:
            return pickle.load(f)


# %%
data_set_name = ['hils_swls_delta', 'score_608'][1]
final_model_test = True
path = '../_data/'
df = pd.read_feather(f'{path}{data_set_name}.feather')
sub_folder = 'final_model_test/' if final_model_test else ''
model_list = [
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "bert-large-cased",
    "roberta-base",
    "roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
    "albert-base-v2",
    "albert-large-v2",
    "distilbert-base-uncased",
    "distilbert-base-cased",
]
mod = [
    'bert-base-uncased',
    'roberta-large',
    'distilbert-base-uncased',
    "roberta-base",
    'bert-large-uncased',
    "xlnet-base-cased"]
# model_name = mod[5]
# %%
for model_name in mod:
    te = torch_embs(
        model_name=model_name,
        device_name='gpu',
        logging_level='warning')
    t = [1]
    scale = ['hils', 'swls']
    for ti in t:
        for s in scale:
            file_name = f'{data_set_name}_{s}_t{ti}_{model_name}_embs_dict'
            word_col = (
                f'harmony_t{ti}'
                if s == 'hils'
                else f'satisfaction_t{ti}')
            embs, toks = te.get_embs(
                text_strings=df[word_col].tolist())
            embs_dict = te.get_embs_dict(
                embs=embs,
                df=df,
                word_col=word_col)
            te.save_embs_dict_to_pickle(
                embs_dict=embs_dict,
                file_name=file_name,
                save_path=f'{path}embs_dicts/{sub_folder}')

# %%
