import json
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

from applications.shared.constants import *
from applications.shared.utilites.text_utilites import *
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizerFast
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")


# device= torch.device("cpu")

def from_json_to_csv(json_file):
    with open(json_file) as json_file:
        out_path = json_file.name.replace(".json", ".xlsx")
        print(out_path)
        j_data = json.load(json_file)
        l_data = list(j_data.values())
        df = pd.DataFrame(l_data)
        # df.drop(['TEMPLATE', 'NERVALS', 'NER_TAGS','ids'],axis=1, inplace=True)
        # roberta now has its own genrator

        # sentence,slots,intent_label
        column_name_mapping = {'TEXT': 'sentence', 'NER_TAGS': 'slots', 'INTENTS': 'intent_label'}
        df = df.rename(columns=column_name_mapping)

        # Specify the desired order of columns
        desired_column_order = ['sentence', 'slots', 'intent_label']

        # Use the loc method to select columns in the desired order
        df = df.loc[:, desired_column_order]
        df['intent_label'] = df['intent_label'].apply(lambda x: x[0] if x else None)
        df['sentence'] = df['sentence'].apply(lambda x: remove_punctuation(" ".join(x)))

        df.to_excel(out_path, index=False)
        return df


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
                label_ids.append(slot2idx[slotLabels[word_idx]])
            previous_word_idx = word_idx

        # x, y_slot, y_intent
        return tokenized_inputs, label_ids

    # def tokenize_dataset_deprecated(self, dataset):
    #     # Tokenize entire dataset
    #     dataset_x = []
    #     dataset_y_slot = []
    #     dataset_y_intent = []
    #
    #     for data in dataset:
    #         x, y_slot, y_intent = self.tokenize_and_align_labels(data, slot2idx, intent2idx)
    #         dataset_x.append(x)
    #         dataset_y_slot.append(y_slot)
    #         dataset_y_intent.append(y_intent)
    #
    #     dataset_x = pd.DataFrame(dataset_x)
    #
    #     dataset_x = {
    #         "input_ids": tf.constant(value=np.stack(dataset_x["input_ids"].values), dtype="int32"),
    #         # "token_type_ids": tf.constant(value=np.stack(dataset_x["token_type_ids"].values), dtype="int32"), # XLM-RoBERTa dosen't need to it
    #         "attention_mask": tf.constant(value=np.stack(dataset_x["attention_mask"].values), dtype="int32"),
    #     }
    #
    #     dataset_y_slot = tf.constant(dataset_y_slot)
    #     dataset_y_intent = tf.constant(dataset_y_intent)
    #
    #     return dataset_x, dataset_y_slot, dataset_y_intent

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


def tokenize_and_align_labels(data, slot2idx, intent2idx):
    # Your implementation here
    pass


def tokenize_dataset(dataset):
    # Your implementation here
    pass


# class JointIntentAndSlotFillingModel(nn.Module):
#     def __init__(self, intent_num_labels=None, slot_num_labels=None,
#                  bert=None, dropout_prob=0.1):
#         super().__init__()
#         #  self.bert = bert
#         # self.bert.training = False
#         self.dropout1 = nn.Dropout(dropout_prob)
#         self.dropout2 = nn.Dropout(dropout_prob)
#
#         self.intent_classifier = nn.Linear(bert.config.hidden_size, intent_num_labels)
#         self.slot_classifier = nn.Linear(bert.config.hidden_size, slot_num_labels)
#
#     def forward(self, input_ids, attention_mask, bert, **kwargs):
#         # self.bert.training = False
#         # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#
#         outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
#
#         pooled_output = outputs.pooler_output
#         sequence_output = outputs.last_hidden_state
#
#         sequence_output = self.dropout1(sequence_output)
#
#         # Slot classification
#         slot_logits = self.slot_classifier(sequence_output)
#
#         # Intent classification
#         pooled_output = self.dropout2(pooled_output)
#         intent_logits = self.intent_classifier(pooled_output)
#
#         return slot_logits, intent_logits

import torch.nn.functional as F

class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 bert=None, dropout_prob=0.1, hidden_size=512, num_hidden_layers=3):
        super().__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout_prob)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(bert.config.hidden_size, hidden_size)
        ])
        # Additional hidden layers
        self.hidden_layers.extend([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_hidden_layers)
        ])

        self.intent_classifier = nn.Linear(hidden_size, intent_num_labels)
        self.slot_classifier = nn.Linear(hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask, bert, **kwargs):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply additional hidden layers
        for layer in self.hidden_layers:
            sequence_output = F.relu(layer(sequence_output))
            sequence_output = self.dropout(sequence_output)

        # Slot classification
        slot_logits = self.slot_classifier(sequence_output)

        # Intent classification
        pooled_output = sequence_output.mean(dim=1)  # Use mean pooling
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits


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


# class CombineCallback:
#     # Your implementation here
#     pass

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
            return True

        return False


def tokenize_test_dataset(dataset):
    # Your implementation here
    pass


def postprocess_evaluation_roberta(tokens, labels, sentence, tokenizer):
    # This function just keep one slot label per word and remove other slot labels
    # In fact, we convert out output format to the challenge output format
    # Note that it developed only for XLM-RoBERTa-Large not other language models

    sentence = sentence.replace("\u200c", "")
    real_tokens = sentence.split()
    id = 0  # Counter for output tokens
    real_id = 0  # Counter for real tokens
    output_labels = []  # Output slot labels

    while (real_id < len(real_tokens)):
        real_token = real_tokens[real_id]
        token = tokens[id]
        token_word = tokenizer.decode(tokens[id])

        if token_word == "<s>" or token_word == "</s>":
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
                token_word = robertaTokenizer.decode(tokens[id])
                last_index = first_index + len(token_word)
                assert (real_token[first_index: last_index] == token_word)
                first_index = last_index
                id += 1
            real_id += 1

    new_output_labels = []
    # Ummm, if a pad token exist, we convert it to O label
    for label in output_labels:
        # if label == "<pad>":
        #     label = "O"
        new_output_labels.append(label)

    return new_output_labels


def postprocess_evaluation_bert(tokens, labels, sentence):
    # Your implementation here
    pass


if __name__ == "__main__":

    df = from_json_to_csv(questions_dirctory + 'q_roberta_0.json')[:-100]

    # Get Labels and  make dictionaries

    # load models
    conf_path = ".storage/XLMRoberta/conf"
    base_path = ".storage/XLMRoberta/base"
    # base_path = '.storage/XLMRoberta/brt.pth'
    tokenizer_path = ".storage/XLMRoberta/tokenizer"
    final_path = ".storage/XLMRoberta/final"

    train_again = False
    num_epochs = 0




    # train_again = True
    # num_epochs = 3

    if train_again:

        train_size = int(0.85 * len(df))
        valid_size = len(df) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(df, [train_size, valid_size])

        # Initialize PyTorch Dataset
        train_dataset = CustomDataset(train_dataset, robertaTokenizer, slot2idx, intent2idx)
        valid_dataset = CustomDataset(valid_dataset, robertaTokenizer, slot2idx, intent2idx)

        # Create PyTorch DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

        slot_loss_fn = CustomNonPaddingTokenLoss(len(ALL_SLOTS_TAGS))
        intent_loss_fn = CustomNonPaddingTokenLoss(len(ALL_INTENT_TAGS))  # nn.CrossEntropyLoss()


        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                # Move inputs and labels to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_label'].to(device)
                slot_labels = batch['slot_labels'].to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                slot_logits, intent_logits = model(input_ids, attention_mask, robertaModel)

                # Calculate losses
                slot_loss = slot_loss_fn(slot_logits, slot_labels)
                intent_loss = intent_loss_fn(intent_logits, intent_labels)

                # Total loss (combine slot and intent losses)
                total_loss = slot_loss + intent_loss

                # Backward pass and optimization step
                total_loss.backward()
                optimizer.step()

            # Validation steps...
            model.eval()  # Set the model to evaluation mode
            val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch in tqdm(valid_dataloader, desc=f"Validation - Epoch {epoch + 1}/{num_epochs}"):
                    # Move inputs and labels to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    intent_labels = batch['intent_label'].to(device)
                    slot_labels = batch['slot_labels'].to(device)

                    # Forward pass
                    slot_logits, intent_logits = model(input_ids, attention_mask, robertaModel)

                    # Calculate losses
                    val_slot_loss += slot_loss_fn(slot_logits, slot_labels).item()
                    val_intent_loss += intent_loss_fn(intent_logits, intent_labels).item()

                    # Calculate accuracies
                    val_slot_accuracy += (torch.argmax(slot_logits, dim=-1) == slot_labels).float().mean().item()
                    val_intent_accuracy += (
                            torch.argmax(intent_logits, dim=-1) == intent_labels).float().mean().item()

                # Average the validation metrics
                val_slot_loss /= len(valid_dataloader)
                val_intent_loss /= len(valid_dataloader)
                val_slot_accuracy /= len(valid_dataloader)
                val_intent_accuracy /= len(valid_dataloader)

                # Callback to check if it's the best epoch and save the model
                if combine_callback.on_epoch_end(epoch, val_slot_loss, val_intent_loss, val_slot_accuracy,
                                                 val_intent_accuracy):
                    # Save the entire model (including its architecture and optimizer state) for future use
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_slot_loss': val_slot_loss,
                        'val_intent_loss': val_intent_loss,
                        'val_slot_accuracy': val_slot_accuracy,
                        'val_intent_accuracy': val_intent_accuracy,
                    }, final_path)

                print(f"Average Validation Slot Loss: {val_slot_loss}")
                print(f"Average Validation Intent Loss: {val_intent_loss}")
                print(f"Average Validation Slot Accuracy: {val_slot_accuracy}")
                print(f"Average Validation Intent Accuracy: {val_intent_accuracy}")

    l1 = list(df[-100:]['sentence'])
    l2 = list(df[-100:]['slots'])
    l3 = list(df[-100:]['intent_label'])

    dst_test = pd.DataFrame({'sentence': l1,
                             'slots': l2,
                             'intent_label': l3})

    # Load test dataset (assuming you have it loaded)
    dst_test_ = CustomDataset(dst_test, robertaTokenizer, slot2idx, intent2idx)

    test_batch = next(iter(DataLoader(dst_test_, batch_size=len(dst_test_))))

    input_ids = test_batch['input_ids'].to(device)
    attention_mask = test_batch['attention_mask'].to(device)

    model.eval()  # Set the model to evaluation mode
    y_test_predict = model(input_ids, attention_mask, robertaModel)  # Assuming model is your trained PyTorch model

    cpu_device = torch.device('cpu')
    input_ids = test_batch['input_ids'].to(cpu_device)
    attention_mask = test_batch['attention_mask'].to(cpu_device)
    intent_labels = test_batch['intent_label'].to(cpu_device)
    slot_labels = test_batch['slot_labels'].to(cpu_device)

    y_slot_test_probabilities = torch.nn.functional.softmax(y_test_predict[0], dim=2)
    y_slot_test_class_preds = torch.argmax(y_slot_test_probabilities, dim=2).cpu().numpy()

    y_intent_test_probabilities = torch.nn.functional.softmax(y_test_predict[1], dim=1)
    y_intent_test_class_preds = torch.argmax(y_intent_test_probabilities, dim=1).cpu().numpy()

    print("y_test_slot_class_preds.shape = ", y_slot_test_class_preds.shape)
    print("y_test_intent_class_preds.shape = ", y_intent_test_class_preds.shape)

    # Postprocess
    test_slots = []
    test_intent_label = []

    for sentence in y_slot_test_class_preds:
        test_slots.append([idx2slot[int(k)] for k in sentence])

    for intent in y_intent_test_class_preds:
        test_intent_label.append(idx2intent[intent])

    # Same format preprocessing for test
    # processed_dataset_test = list()
    # for i in range(len(l1)):
    #     slot_tokens_test = []
    #
    #     for token in l1[i].split():
    #         slot_tokens_test.append(token)
    #
    #     processed_dataset_test.append({"input": l1[i],
    #                                    "slot_tokens": slot_tokens_test})
    #
    # dataset_test = pd.DataFrame(processed_dataset_test)
    # print(dataset_test)
    #
    # test_dataset_hug = Dataset.from_pandas(dataset_test)
    # print(test_dataset_hug)

    df_test = pd.DataFrame(list(zip(l1, test_slots, test_intent_label)),
                           columns=['sentence', 'slots', 'intent_label'])
    print(df_test.head())

    # Postprocess evaluation
    print(df_test['sentence'][1])
    print([i for i in df_test['slots'][1] if i != '<PAD>'])
    print([i for i in df_test['slots'][1] if i != '<pad>'])

    print(df_test['intent_label'][1])

    new_slots = []
    for i in range(len(l1)):
        new_slots.append(
            postprocess_evaluation_roberta(input_ids.numpy()[i], df_test['slots'].iloc[i], l1[i], robertaTokenizer))

    # Check that the length of slot label list is the same as the number of tokens
    count = 0
    for ii in range(len(df_test)):
        if len(df_test['sentence'][ii].split()) != len(
                postprocess_evaluation_roberta(input_ids.numpy()[ii], df_test['slots'].iloc[ii], l1[ii],robertaTokenizer)):
            print(ii)
            print(df_test['sentence'][ii])
            print(len(df_test['sentence'][ii].split()))

            print(
                postprocess_evaluation_roberta(input_ids.numpy()[ii], df_test['slots'].iloc[ii],
                                               l1[ii]))
            print(
                len(postprocess_evaluation_roberta(input_ids.numpy()[ii], df_test['slots'].iloc[ii],
                                                   l1[ii])))

            count += 1

    print("Number of unmatched length: ", count)

    # Output Predictions
    df_test['slots'] = new_slots
    df_test['slots'] = [' '.join(i) for i in df_test['slots']]
    df_test.reset_index(inplace=True)

    df_test.drop(['sentence'], axis=1).to_csv('./predictions.csv', index=False)
    df_test.to_csv('./predictions.csv', index=False)
    df_test.to_excel('./predictions.xlsx', index=False)

    print("Hope you enjoy spending some time with us :)")

    # Example: Inference using the trained model on a new input
    new_input_text = "Your new input text here"
    tokenized_input = robertaTokenizer(new_input_text, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        slot_logits, intent_logits = model(input_ids, attention_mask, robertaModel)

    # Process the logits as needed for your specific task (e.g., getting predictions)

    # Additional code as needed based on your requirements

    # Rest of the code...

    exit()

    # Same format preprocessing for test
    processed_dataset_test = list()
    for i in range(len(x_test)):
        slot_tokens_test = []

        for token in x_test[i].split():
            slot_tokens_test.append(token)

        processed_dataset_test.append({"input": x_test[i],
                                       "slot_tokens": slot_tokens_test})

    dataset_test = pd.DataFrame(processed_dataset_test)
    print(dataset_test)

    test_dataset_hug = Dataset.from_pandas(dataset_test)
    print(test_dataset_hug)
