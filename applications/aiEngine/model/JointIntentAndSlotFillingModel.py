import os
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import AdamW
#from transformers import XLMRobertaModel#, XLMRobertaTokenizer, XLMRobertaTokenizerFast as OurTokenizer  # Fast
from transformers import AutoModel as OurModel, AutoTokenizer as OurTokenizer
from shared.constants.constants import *
from applications.aiEngine.utilities import get_Slots_Intentes


class CombineCallback:
    def __init__(self, model, save_path, best=0):
        self.model = model
        self.best_metric = best
        self.save_path = save_path

    def on_epoch_end(self, epoch, val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy):
        # Your logic to combine metrics (if needed)
        combine_metric = val_slot_accuracy + val_intent_accuracy

        if combine_metric > self.best_metric:
            self.best_metric = combine_metric
            print(f"saved with {val_slot_accuracy} {val_intent_accuracy}")
            self.model.save_model_as_is(epoch, val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy)
            return True

        return False


class CustomNonPaddingTokenLoss(nn.Module):
    def __init__(self, num_class, ignore_index=None):
        super(CustomNonPaddingTokenLoss, self).__init__()
        #self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the index for <PAD>
        if ignore_index is not None:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)  # Assuming 0 is the index for <PAD>
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        # mask = (y_true > 0).float()
        # loss = self.loss_fn(y_pred, y_true) #* mask
        # loss = self.loss_fn(y_pred.view(0,-1), y_true.view(-1, 1))
        # torch.flatten(y_true,end_dim=-2)
        # loss = self.loss_fn(y_pred.view(-1, self.num_class), y_true.view(-1)) * mask.view(-1)

        # return torch.sum(loss) / torch.sum(mask)

        return self.loss_fn(y_pred.view(-1, self.num_class), y_true.view(-1)).mean()

class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self,
                 bert=None, method='light', dropout_prob=0.1, hidden_size=512, num_hidden_layers=3):
        super(JointIntentAndSlotFillingModel, self).__init__()

        self.pre_trained_model = True
        self.method = method
        self.optimizer = None
        self.combine_callback = None
        self.robertaTokenizer = None
        self.robertaModel = None

        self.slot2idx, self.idx2slot, self.intent2idx, self.idx2intent = get_Slots_Intentes()

        self.bert = bert
        self.dropout = nn.Dropout(dropout_prob)

        self.slots_hidden_layers = nn.ModuleList([
            nn.Linear(bert.config.hidden_size, hidden_size)
        ])
        # Additional hidden layers
        self.slots_hidden_layers.extend([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_hidden_layers)
        ])

        self.slot_classifier = nn.Linear(hidden_size, len(self.slot2idx))

        self.intent_hidden_layers = nn.ModuleList([
            nn.Linear(bert.config.hidden_size, hidden_size)
        ])
        # Additional hidden layers
        self.intent_hidden_layers.extend([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_hidden_layers)
        ])

        self.intent_classifier = nn.Linear(hidden_size, len(self.intent2idx))

        # self.load_and_save_pretrained_model()

    def forward(self, input_ids, attention_mask, bert = None, **kwargs):
        bert = bert or self.bert
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_out = outputs.pooler_output

        # Apply additional hidden layers
        for layer in self.slots_hidden_layers:
            sequence_output = F.relu(layer(sequence_output))
            sequence_output = self.dropout(sequence_output)

        # Slot classification
        slot_logits = self.slot_classifier(sequence_output)

        # Intent classification
        pooled_output = pooled_out

        for layer in self.intent_hidden_layers:
            pooled_output = F.relu(layer(pooled_output))
            pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits

    def state_dict(self, destination=None, prefix='', keep_vars=False, **kwargs):

        state_dict = super(JointIntentAndSlotFillingModel, self).state_dict(destination, prefix, keep_vars)

        # Modify state_dict if needed, e.g., remove specific layers
        # For example, let's exclude the parameters of fc2
        for key in list(state_dict.keys()):
            if 'bert' in key and self.method == "light":
                del state_dict[key]

        return state_dict

    def parameters(self, **kwargs):
        target_parameters = []
        for name, param in self.named_parameters(**kwargs):
            if 'bert' not in name or self.method != "light":
                target_parameters.append(param)
        return target_parameters

    def save_model_as_is(self, epoch, val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy):
        if hasattr(self, 'robertaModel'):
            self.robertaModel.save_pretrained(xlm_mixed_path)

        if hasattr(self, 'robertaTokenizer'):
            self.robertaTokenizer.save_pretrained(xlm_mixed_path)
        # Save the entire model (including its architecture and optimizer state) for future use
        torch.save({
            'epoch': 0,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_slot_loss': np.inf,
            'val_intent_loss': np.inf,
            'val_slot_accuracy': .0,
            'val_intent_accuracy': .0,
        }, final_path)

    def load_and_save_pretrained_model(self):

        if not os.path.exists(final_path):
            self.pre_trained_model = False
            self.robertaModel = OurModel.from_pretrained(model_name)
            self.robertaTokenizer = OurTokenizer.from_pretrained(model_name)
            self.optimizer = AdamW(self.parameters(), lr=3e-5)
            self.save_model_as_is(0, .0, .0, .0, .0)
            print("Model Must be trained")

        self.robertaModel = OurModel.from_pretrained(xlm_mixed_path)
        self.robertaTokenizer = OurTokenizer.from_pretrained(xlm_mixed_path)

        values = torch.load(final_path, map_location=device)  # Initialize PyTorch Model
        self.load_state_dict(values['model_state_dict'], strict=False)
        self.to(device)
        self.robertaModel.to(device)

        self.optimizer = AdamW(self.parameters(), lr=3e-3)
        self.optimizer.load_state_dict(values['optimizer_state_dict'])

        self.loss = {"slots": None , "intents": None }
        self.loss["slots"] = CustomNonPaddingTokenLoss(len(self.slot2idx), 0)
        self.loss["slots"] = CustomNonPaddingTokenLoss(len(self.slot2idx))
        self.loss["intent"] = CustomNonPaddingTokenLoss(len(self.intent2idx))

        self.combine_callback = CombineCallback(self, final_path,
                                                values['val_slot_accuracy'] + values['val_intent_accuracy'])

