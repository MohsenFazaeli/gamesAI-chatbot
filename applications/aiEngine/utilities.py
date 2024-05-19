import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# from shared import *
# from transformers import AutoTokenizer#, AutoModelForSequenceClassification


# import tensorflow as tf
# import tensorflow_addons as tfa
#
# import torch
# from torch.utils.data import Dataset, DataLoader
#
# from shared import *
# from transformers import AutoTokenizer#, AutoModelForSequenceClassification
#
# # tf.config.run_functions_eagerly(True)
# # TODO decide eagarly
#
# @tf.function
# def macro_soft_f1(y, y_hat):
#     """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
#     Use probability values instead of binary predictions.
#
#     Args:
#         y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
#
#     Returns:
#         cost (scalar Tensor): value of the cost function for the batch
#     """
#     y = tf.cast(y, tf.float32)
#     y_hat = tf.cast(y_hat, tf.float32)
#     tp = tf.reduce_sum(y_hat * y, axis=0)
#     fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
#     fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
#     soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
#     cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
#     macro_cost = tf.reduce_mean(cost)  # average on all labels
#     return macro_cost
#
#
# @tf.function
# def macro_f1(y, y_hat, thresh=0.5):
#     """Compute the macro F1-score on a batch of observations (average F1 across labels)
#
#     Args:
#         y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
#         thresh: probability value above which we predict positive
#
#     Returns:
#         macro_f1 (scalar Tensor): value of macro F1 for the batch
#     """
#     y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
#     tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
#     fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
#     fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
#     f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
#     macro_f1 = tf.reduce_mean(f1)
#     return macro_f1
#
#
# class MacroF1(tf.keras.metrics.Metric):
#
#     def __init__(self, name='MacroF1', **kwargs):
#         super(MacroF1, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name='tp', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         thresh = 0.5
#         y_hat = tf.cast(tf.reshape(y_pred, [-1, 11]), tf.float32)
#
#         y2 = tf.cast(y_true, tf.float32)
#         y2 = tf.keras.utils.to_categorical(y2, num_classes=11)
#         y2 = tf.reshape(y2, [-1])
#         # oh_labels = tf.cast(tf.one_hot(tf.cast(labels, tf.int32), num_classes), dtype=labels.dtype)
#         # print(y)
#         # y2 = tf.cast(tf.one_hot(tf.cast(y2, tf.int32), 11), dtype=tf.int32)
#         y = tf.cast(y2, tf.float32)
#
#         print(y.shape, y_hat)
#         print(y.shape, y_hat)
#         print(y.shape, y_hat)
#         print(y.shape, y_hat)
#         print(y.shape, y_hat)
#         print(y.shape, y_hat)
#
#         y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
#         tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
#         fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
#         fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
#         f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
#         macro_f1 = tf.reduce_mean(f1)
#
#     def result(self):
#         return self.true_positives
#
#
# class MY_F1Score(tfa.metrics.FBetaScore):
#     def __init__(
#             self,
#             num_classes: 11,
#             average=None,
#             threshold=None,
#             name="MY_F1Score",
#             dtype=None,
#     ):
#         super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)
#
#     def get_config(self):
#         base_config = super().get_config()
#         del base_config["beta"]
#         return base_config
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # print(y_true,y_pred)
#         # y_true_ = tf.keras.utils.to_categorical(y_true, num_classes=11)
#         y_true_ = tf.constant(tf.reshape(y_true, [-1, y_true.shape[-1]]))
#         y_pred_ = tf.constant(tf.reshape(y_pred, [-1, y_pred.shape[-1]]))
#         # print(y_true_,y_pred_)
#
#         super().update_state(y_true_, y_pred_, sample_weight=None)


# from shared.constants.constants import *
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, number_of_questions, INTENTS_MAP_reverse, SLOTS_MAP_reverse, getSparse=False):
#         self.batch_size = BATCH_SIZE
#         self.num_batches = int(number_of_questions / BATCH_SIZE)
#         self.current_file = 0
#         self.number_of_questions = number_of_questions
#         self.load_json()
#         self.INTENTS_MAP_reverse = INTENTS_MAP_reverse
#         self.len_intnets = len(self.INTENTS_MAP_reverse)
#         self.SLOTS_MAP_reverse = SLOTS_MAP_reverse
#         self.len_slots = len(self.SLOTS_MAP_reverse)
#         self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", padding=True, pad_token="[PAD]")
#         self.getSparse = getSparse
#
#     def load_json(self):
#         with open(questions_dirctory + 'q_' + str(self.current_file) + '.json', 'r', encoding='utf-8') as fp:
#             self.data = json.load(fp)
#
#     def __len__(self):
#         return self.num_batches
#
#     def __getitem__(self, index):
#         # Implement logic to load and return a batch of data
#         #print("  ", index)
#         x_batch, y_batch1, y_batch2 = self.load_next_batch(index)
#         return x_batch, {'intents': y_batch1, 'slots': y_batch2 }
#
#     def on_epoch_end(self):
#         # Implement logic to run at the end of each epoch if needed
#         pass
#
#     def load_next_batch(self, idx):
#         # Implement logic to load and return a batch of data
#         # This method should be tailored to your specific data loading needs
#         idx = idx * self.batch_size
#         file_number = int(idx/number_of_questions_per_file)
#         if file_number != self.current_file:
#             self.current_file = file_number
#             self.load_json()
#         infile_idx = idx - file_number*number_of_questions_per_file
#         data = self.data
#         current_batch = []
#         if number_of_questions_per_file > infile_idx + self.batch_size:
#             for i in range(infile_idx, infile_idx + self.batch_size):
#                 current_batch.append(data[str(i)])
#         elif idx > self.number_of_questions - self.batch_size:
#             for i in range(infile_idx, number_of_questions_per_file):
#                 current_batch.append(data[str(i)])
#         else:
#             for i in range(infile_idx, number_of_questions_per_file):
#                 #print(self.current_file, i)
#
#                 current_batch.append(data[str(i)])
#             self.current_file += 1
#             self.load_json()
#             data = self.data
#             for i in range(0, self.batch_size - number_of_questions_per_file + infile_idx):
#                 current_batch.append(data[str(i)])
#
#         x_batch, y_batch1, y_batch2 = self.get_batch(current_batch)
#         return x_batch, y_batch1, y_batch2
#
#     def get_batch(self, current_batch):
#         y_batch1, y_batch2 = [], []
#         x_batch = np.zeros(shape=(self.batch_size, MAX_LENTGH), dtype=np.int32)
#         for i,value in enumerate(current_batch):
#             x_batch[i,0:len(value['ids'])] = value['ids'] #value['ids']
#             y1 = self.INTENTS_MAP_reverse[value['INTENTS'][0]]
#             y2 = value['NER_TAGS']
#             y_batch1.append(y1)
#             y_batch2.append(y2)
#
#         y_batch2 = self.encode_slots_label(y_batch2)
#
#         if not self.getSparse:
#             y_batch1 = self.embed_intent(y_batch1)
#             y_batch2 = self.embed_slots(y_batch2)
#
#         return np.array(x_batch), np.array(y_batch1), np.array(y_batch2)
#
#     def encode_slots_label(self, y_batch2, max_length=MAX_LENTGH):
#         encoded = np.zeros(shape=(len(y_batch2), max_length), dtype=np.int32)
#         for i, labels in enumerate(y_batch2):
#             encoded_labels = [self.SLOTS_MAP_reverse[label] for label in labels]
#             encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
#         return encoded
#
#     def embed_intent(self, intents):
#         #return np.array(tf.keras.utils.to_categorical(self.INTENTS_MAP_reverse[intent[0]], num_classes=self.len_intnets))
#         return np.array(tf.keras.utils.to_categorical(intents, num_classes=self.len_intnets))
#
#     def embed_slots(self, slots):
#         return np.array(tf.keras.utils.to_categorical(slots, num_classes=self.len_slots))


def get_Slots_Intentes():
    from shared.constants.constants import question_temps

    ALL_SLOTS_TAGS = []
    ALL_INTENT_TAGS = []
    for item in question_temps.values():
        ALL_SLOTS_TAGS.extend(item['NERTAGS'])
        ALL_INTENT_TAGS.extend(item['INTENTS'])
        bad_tags = [i for i in item['NERTAGS'] if not (i.endswith('_1') or i.endswith("_2") or i == 'O')]
        if len(bad_tags) > 0:
            print("Bad tags: ", bad_tags)
            print(item)
            exit()

    for tag in ALL_SLOTS_TAGS:
        if tag.startswith('B-'):
            ALL_SLOTS_TAGS.append(tag.replace('B-', 'I-'))

    tmp = sorted(set(ALL_SLOTS_TAGS))
    tmp.remove('O')
    tmp.sort()
    print("a",tmp)
    ALL_SLOTS_TAGS = ["<pad>"] + tmp + ['O']
    print(len(ALL_SLOTS_TAGS), ALL_SLOTS_TAGS)

    idx2slot = {i: v for i, v in enumerate(ALL_SLOTS_TAGS)}
    slot2idx = {v: i for i, v in enumerate(ALL_SLOTS_TAGS)}

    ALL_INTENT_TAGS = sorted(set(ALL_INTENT_TAGS))
    print(len(ALL_INTENT_TAGS), ALL_INTENT_TAGS)

    idx2intent = {i: v for i, v in enumerate(ALL_INTENT_TAGS)}
    intent2idx = {v: i for i, v in enumerate(ALL_INTENT_TAGS)}

    return slot2idx, idx2slot, intent2idx, idx2intent


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, slot2idx, intent2idx):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.slot2idx = slot2idx
        self.intent2idx = intent2idx

    def __len__(self):
        return len(self.data)

    def tokenize_and_align_labels(self, tokens, slotLabels):
        # Tokenise a data and aligning with slot labels
        # If a word splited to some subwords, we must handle slot label

        tokenized_inputs = self.tokenizer(tokens,
                                          truncation=True, is_split_into_words=True,
                                          padding="max_length", max_length=30, return_tensors="pt")

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []

        # Set the special tokens to 0.
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(0)
            elif word_idx == previous_word_idx:
                # Pad label for other tokens of a word
                label_ids.append(0)
            else:
                # Only label the first token of a given word.
                label_ids.append(self.slot2idx[slotLabels[word_idx]])
            previous_word_idx = word_idx

        # x, y_slot, y_intent
        return tokenized_inputs, label_ids

    def __getitem__(self, idx):
        try:
            data = self.data.iloc[idx]
        except:
            real_idx = self.data.indices[idx]
            data = self.data.dataset.iloc[real_idx]

        text = data['sentence']
        slot_labels = data['slots']
        tokens = text.split()
        tokenized, labels = self.tokenize_and_align_labels(tokens, slot_labels)
        intent_label = data['intent_label']

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'slot_labels': torch.tensor(labels, dtype=torch.long),
            'intent_label': torch.tensor(self.intent2idx[intent_label], dtype=torch.long)
        }

import torch.nn as nn

class CustomNonPaddingTokenLoss(nn.Module):
    def __init__(self, num_class):
        super(CustomNonPaddingTokenLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the index for <PAD>
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')  # Assuming 0 is the index for <PAD>
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        mask = (y_true > 0).float()
        # loss = self.loss_fn(y_pred, y_true) #* mask
        # loss = self.loss_fn(y_pred.view(0,-1), y_true.view(-1, 1))
        # torch.flatten(y_true,end_dim=-2)
        loss = self.loss_fn(y_pred.view(-1, self.num_class), y_true.view(-1)) * mask.view(-1)

        return torch.sum(loss) / torch.sum(mask)

# Assume you have a model and compile it as before
# ...

# # Create an instance of the DataGenerator
# batch_size = 32
# num_batches = 1000
# data_generator = DataGenerator(batch_size, num_batches)
#
# # Train the model using fit_generator
# model.fit(data_generator, epochs=5)
if __name__ == "__main__":
    from shared import *
    from applications.aiEngine.train.preprocess import *

    dGen = DataGenerator(number_of_questions, INTENTS_MAP_reverse, SLOTS_MAP_reverse)
    # x_batch, a = dGen.__getitem__(10)
    # print(x_batch[0])
    # print(x_batch.shape,x_batch)
    # print(a['intents'].shape,a['intents'])
    # print(a['slots'].shape,a['slots'])
    #
    print(dGen.__len__())
    for batch in dGen:
        a, b = batch
        print(a.shape,b['intents'].shape,b['slots'].shape)

