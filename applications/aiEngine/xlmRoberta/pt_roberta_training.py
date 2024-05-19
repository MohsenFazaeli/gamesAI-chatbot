import json
import os
from torch.utils.data import Dataset

import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from shared.constants.constants import *
from applications.utilites.text_utilites import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

#device= torch.device("cpu")

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

    def tokenize_and_align_labels(slef, tokens, slotLabels):
        # Tokenise a data and aligning with slot labels
        # If a word splited to some subwords, we must handle slot label

        tokenized_inputs = tokenizer(tokens,
                                     truncation=True, is_split_into_words=True,
                                     padding="max_length", max_length=30, return_tensors="pt")

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []

        # Set the special tokens to 0.-
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
        real_idx = self.data.indices[idx]
        text = self.data.dataset['sentence'].iloc[real_idx]
        slot_labels = [slot for slot in self.data.dataset['slots'].iloc[real_idx]]
        tokens = text.split()
        tokenized, labels = self.tokenize_and_align_labels(tokens, slot_labels)
        intent_label = self.data.dataset['intent_label'].iloc[
            real_idx]  # self.intent2idx[self.data['intent_label'].iloc[idx]]

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


class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name=model_name, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_num_labels)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        # Slot classification
        slot_logits = self.slot_classifier(sequence_output)

        # Intent classification
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
        loss = self.loss_fn(y_pred.view(-1, self.num_class), y_true(-1)) * mask.view(-1)

        return torch.sum(loss) / torch.sum(mask)


# class CombineCallback:
#     # Your implementation here
#     pass

class CombineCallback:
    def __init__(self, model, save_path):
        self.model = model
        self.best_metric = 0.0
        self.save_path = save_path

    def on_epoch_end(self, epoch, val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy):
        # Your logic to combine metrics (if needed)
        combine_metric = val_slot_accuracy + val_intent_accuracy

        if combine_metric > self.best_metric:
            self.best_metric = combine_metric
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))


def tokenize_test_dataset(dataset):
    # Your implementation here
    pass


def postprocess_evaluation_roberta(tokens, labels, sentence):
    # Your implementation here
    pass


def postprocess_evaluation_bert(tokens, labels, sentence):
    # Your implementation here
    pass


if __name__ == "__main__":

    df = from_json_to_csv(questions_dirctory + 'q_roberta_0.json')
    # df_test = pd.read_csv("dataset/test.csv")
    # TODO impliment a test xlsx
    # df_test = pd.read_csv("dataset/test.csv")
    x_test = df['sentence'][-100:].reset_index(drop=True)
    #df = df[:100]

    # Check that if number of sentences is equal to its slots number or not
    for i in range(len(df)):
        if len(df['sentence'][i].split()) != len(df['slots'][i]):
            # So we have a bad data
            print(i)
            print(df['sentence'][i].split())
            print(df['slots'][i].split())

    text_dataset = df['sentence']
    intent_label_dataset = df['intent_label']

    # Check all exist intens
    set(intent_label_dataset)

    slot_label_dataset = df['slots']
    # slot_label_dataset = [change_slot_preprocess(i) for i in slot_label_dataset]

    slot_label_dataset = slot_label_dataset.tolist()

    #from shared import question_temps

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
    ALL_SLOTS_TAGS = ["<PAD>"] + tmp + ['O']
    print(len(ALL_SLOTS_TAGS), ALL_SLOTS_TAGS)

    idx2slot = {i: v for i, v in enumerate(ALL_SLOTS_TAGS)}
    slot2idx = {v: i for i, v in enumerate(ALL_SLOTS_TAGS)}

    ALL_INTENT_TAGS = set(sorted(set(ALL_INTENT_TAGS)))
    print(len(ALL_INTENT_TAGS), ALL_INTENT_TAGS)

    idx2intent = {i: v for i, v in enumerate(ALL_INTENT_TAGS)}
    intent2idx = {v: i for i, v in enumerate(ALL_INTENT_TAGS)}

    # Apply some processes for a better formet
    processed_dataset = list()
    for i in range(len(intent_label_dataset)):
        slot_tokens = []
        slots_labels = []

        for token, token_label in zip(text_dataset[i].split(" "), slot_label_dataset[i]):
            slot_tokens.append(token)
            slots_labels.append(token_label)

        processed_dataset.append({"input": text_dataset[i],
                                  "intent": intent_label_dataset[i],
                                  "slot_tokens": slot_tokens,
                                  "slot_labels": slots_labels})

    dataset = pd.DataFrame(processed_dataset)

    print(dataset.head())
    print(dataset.columns)

    # Split the dataset into train and validation
    train_size = int(0.85 * len(df))
    valid_size = len(df) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(df, [train_size, valid_size])

    # Create tokenizers, slot2idx, intent2idx

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize PyTorch Dataset
    train_dataset = CustomDataset(train_dataset, tokenizer, slot2idx, intent2idx)
    valid_dataset = CustomDataset(valid_dataset, tokenizer, slot2idx, intent2idx)

    # Create PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    # Initialize PyTorch Model
    model = JointIntentAndSlotFillingModel(
        intent_num_labels=len(intent2idx), slot_num_labels=len(slot2idx)).to(device)

    # Using Adam as the optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)


    # Define loss and metrics
    slot_loss_fn = CustomNonPaddingTokenLoss(len(ALL_SLOTS_TAGS))
    intent_loss_fn = CustomNonPaddingTokenLoss(len(ALL_INTENT_TAGS))  # nn.CrossEntropyLoss()
    #metrics = [nn.CrossEntropyLoss()]


    # Define loss and metrics
    slot_loss_fn_base = nn.CrossEntropyLoss()
    intent_loss_fn_base = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()
    #metrics = [nn.CrossEntropyLoss()]

    slot_loss_fn = lambda y_pred, y_true : slot_loss_fn_base(y_pred.view(-1,len(ALL_SLOTS_TAGS)), y_true.view(-1))
    intent_loss_fn = lambda y_pred, y_true : slot_loss_fn_base(y_pred.view(-1,len(ALL_INTENT_TAGS)), y_true.view(-1))


    # Your training loop
    num_epochs = 1
    save_path = "./.storage/roberta/"
    # checkpoint_value = torch.load(os.path.join(save_path, 'best_model.pth'))
    # model.load_state_dict(checkpoint_value)

    combine_callback = CombineCallback(model, save_path)

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
            slot_logits, intent_logits = model(input_ids, attention_mask)

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
                slot_logits, intent_logits = model(input_ids, attention_mask)

                # Calculate losses
                val_slot_loss += slot_loss_fn(slot_logits, slot_labels).item()
                val_intent_loss += intent_loss_fn(intent_logits, intent_labels).item()

                # Calculate accuracies
                val_slot_accuracy += (torch.argmax(slot_logits, dim=-1) == slot_labels).float().mean().item()
                val_intent_accuracy += (torch.argmax(intent_logits, dim=-1) == intent_labels).float().mean().item()

        # Average the validation metrics
        val_slot_loss /= len(valid_dataloader)
        val_intent_loss /= len(valid_dataloader)
        val_slot_accuracy /= len(valid_dataloader)
        val_intent_accuracy /= len(valid_dataloader)

        # Callback to check if it's the best epoch and save the model
        combine_callback.on_epoch_end(epoch, val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy)

        print(f"Average Validation Slot Loss: {val_slot_loss}")
        print(f"Average Validation Intent Loss: {val_intent_loss}")
        print(f"Average Validation Slot Accuracy: {val_slot_accuracy}")
        print(f"Average Validation Intent Accuracy: {val_intent_accuracy}")


        # Save the entire model (including its architecture and optimizer state) for future use
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_slot_loss': val_slot_loss,
            'val_intent_loss': val_intent_loss,
            'val_slot_accuracy': val_slot_accuracy,
            'val_intent_accuracy': val_intent_accuracy,
        }, os.path.join(save_path, 'full_model.pth'))

    # Load test dataset (assuming you have it loaded)
    test_dataset_x = tokenize_test_dataset(test_dataset_hug, tokenizer)

    # Predictions
    model.eval()  # Set the model to evaluation mode

    y_test_predict = model(**test_dataset_x)  # Assuming model is your trained PyTorch model

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

    df_test = pd.DataFrame(list(zip(x_test, test_slots, test_intent_label)),
                           columns=['sentence', 'slots', 'intent_label'])
    print(df_test.head())

    # Postprocess evaluation
    print(df_test['sentence'][1])
    print([i for i in df_test['slots'][1] if i != '<PAD>'])

    new_slots = []
    for i in range(len(x_test)):
        new_slots.append(
            postprocess_evaluation_roberta(test_dataset_x["input_ids"].numpy()[i], df_test['slots'].iloc[i], x_test[i]))

    # Check that the length of slot label list is the same as the number of tokens
    count = 0
    for ii in range(len(df_test)):
        if len(df_test['sentence'][ii].split()) != len(
                postprocess_evaluation_roberta(test_dataset_x["input_ids"].numpy()[ii], df_test['slots'].iloc[ii], x_test[ii])):
            print(ii)
            print(df_test['sentence'][ii])
            print(len(df_test['sentence'][ii].split()))

            print(
                postprocess_evaluation_roberta(test_dataset_x["input_ids"].numpy()[ii], df_test['slots'].iloc[ii], x_test[ii]))
            print(
                len(postprocess_evaluation_roberta(test_dataset_x["input_ids"].numpy()[ii], df_test['slots'].iloc[ii], x_test[ii])))

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
    tokenized_input = tokenizer(new_input_text, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        slot_logits, intent_logits = model(input_ids, attention_mask)

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
