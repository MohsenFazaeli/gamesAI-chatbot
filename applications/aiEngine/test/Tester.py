import os
import numpy as np
import torch
from torch.optim import AdamW
# from transformers import XLMRobertaModel, XLMRobertaTokenizer as OurTokenizer # Fast
from tqdm import tqdm

from shared.constants.constants import *
from applications.aiEngine.utilities import CustomDataset, CustomNonPaddingTokenLoss
from torch.utils.data import Dataset, DataLoader


class Tester:
    def __init__(self, data_frame, jmdl):
        self.data_frame = data_frame
        self.model = jmdl

    def test(self):

        # Initialize PyTorch Dataset
        test_dataset = CustomDataset(self.data_frame, self.model.robertaTokenizer, self.model.slot2idx,
                                     self.model.intent2idx)

        # Create PyTorch DataLoader
        tsdl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Test steps...
        self.model.eval()  # Set the model to evaluation mode
        test_slot_accuracy, test_intent_accuracy = 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(tsdl, desc=f"Test"):
                # Move inputs and labels to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_label'].to(device)
                slot_labels = batch['slot_labels'].to(device)

                # Forward pass
                slot_logits, intent_logits = self.model(input_ids, attention_mask)

                # Calculate accuracies
                test_slot_accuracy += (torch.argmax(slot_logits, dim=-1) == slot_labels).float().mean().item()
                test_intent_accuracy += (
                        torch.argmax(intent_logits, dim=-1) == intent_labels).float().mean().item()

            # Average the validation metrics
            test_slot_accuracy /= len(tsdl)
            test_intent_accuracy /= len(tsdl)

            print(f"\nAverage Validation Slot Accuracy: {test_slot_accuracy}")
            print(f"Average Validation Intent Accuracy: {test_intent_accuracy}")

    def tokenize_test_dataset(self, dataset):
        # This function tokenize entire test dataset

        x = self.model.robertaTokenizer(dataset,
                                        truncation=True, is_split_into_words=True,
                                        padding="max_length", max_length=30, return_tensors="pt")

        return x

    def postproces_evaluation_roberta(self, tokens, labels, sentence):

        tokenizer = self.model.robertaTokenizer
        # This function just keep one slot label per word and remove other slot labels
        # In fact, we convert out output format to the challenge output format
        # Note that it developed only for ParsBERT not other language models
        sentence = sentence.replace("\u200c", "")
        real_tokens = sentence.split()

        id = 0  # Counter for output tokens
        real_id = 0  # Counter for real tokens
        output_labels = []  # Output slot labels

        while (real_id < len(real_tokens)):
            real_token = real_tokens[real_id]
            token = tokens[id]
            token_word = tokenizer.decode(tokens[id])
            if "##" == token_word[:2]:
                # Remove non-real chars
                token_word = token_word[2:]

            if token_word in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                # Remove the special label tokens
                id += 1
            elif real_token == token_word:
                # The word split to exact one token
                output_labels.append(labels[id])
                id += 1
                real_id += 1
            else:

                # The word split to more than one token
                # Keep the label of the first token
                first_index = 0
                output_labels.append(labels[id])

                while (first_index < len(real_token)):

                    # Throw away other labels
                    token = tokens[id]
                    token_word = tokenizer.decode(tokens[id])
                    if token_word == "[ZWNJ]":
                        # Remove nim-faselh
                        id += 1
                        continue

                    if "##" == token_word[:2]:
                        # Remove non-real chars
                        token_word = token_word[2:]
                    last_index = first_index + len(token_word)
                    assert (real_token[first_index: last_index] == token_word)
                    first_index = last_index
                    id += 1
                real_id += 1

        new_output_labels = []
        # Ummm, if a pad token exist, we convert it to O label
        for label in output_labels:
            if label == "<PAD>":
                label = "O"
            new_output_labels.append(label)

        return new_output_labels

    def postproces_evaluation_bert(self, tokens, labels, sentence):
        # This function just keep one slot label per word and remove other slot labels
        # In fact, we convert out output format to the challenge output format
        # Note that it developed only for ParsBERT not other language models
        sentence = sentence.replace("\u200c", "")
        real_tokens = sentence.split()

        id = 0  # Counter for output tokens
        real_id = 0  # Counter for real tokens
        output_labels = []  # Output slot labels

        while (real_id < len(real_tokens)):
            real_token = real_tokens[real_id]
            token = tokens[id]
            token_word = tokenizer.decode(tokens[id])
            if "##" == token_word[:2]:
                # Remove non-real chars
                token_word = token_word[2:]

            if token_word in ['[CLS]', '[SEP]', '[PAD]']:
                # Remove the special label tokens
                id += 1
            elif real_token == token_word:
                # The word split to exact one token
                output_labels.append(labels[id])
                id += 1
                real_id += 1
            else:

                # The word split to more than one token
                # Keep the label of the first token
                first_index = 0
                output_labels.append(labels[id])

                while (first_index < len(real_token)):

                    # Throw away other labels
                    token = tokens[id]
                    token_word = tokenizer.decode(tokens[id])
                    if token_word == "[ZWNJ]":
                        # Remove nim-faselh
                        id += 1
                        continue

                    if "##" == token_word[:2]:
                        # Remove non-real chars
                        token_word = token_word[2:]
                    last_index = first_index + len(token_word)
                    assert (real_token[first_index: last_index] == token_word)
                    first_index = last_index
                    id += 1
                real_id += 1

        new_output_labels = []
        # Ummm, if a pad token exist, we convert it to O label
        for label in output_labels:
            if label == "<PAD>":
                label = "O"
            new_output_labels.append(label)

        return new_output_labels

    def nlu(self, input):
        tokenized = self.tokenize_test_dataset([input])
        y_predict = self.model(**tokenized.to(device))  # Assuming model is your trained PyTorch model

        # y_slot_probs = torch.nn.functional.softmax(y_predict[0], dim=2)
        y_slot_preds = torch.argmax(y_predict[0], dim=2).cpu().numpy()
        slots_name = [self.model.idx2slot[int(k)] for k in y_slot_preds[0]]
        slots_name = self.postproces_evaluation_roberta(tokenized["input_ids"].cpu().numpy()[0], slots_name, input)

        params = {}
        for slot, token in zip(slots_name, input.split()):
            if slot == 'O':
                continue

            slot_id = slot.replace('B-', '').replace('I-', '')
            if slot_id in params:
                params[slot_id] += ' ' + token
            else:
                params[slot_id] = token

        # y_intent_test_probabilities = torch.nn.functional.softmax(y_predict[1], dim=1)
        y_intent = torch.argmax(y_predict[1], dim=1).cpu().numpy()
        return params, self.model.idx2intent[y_intent[0]]
