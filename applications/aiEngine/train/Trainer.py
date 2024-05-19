import os
import numpy as np
import torch
from torch.optim import AdamW
#from transformers import XLMRobertaModel, XLMRobertaTokenizer as OurTokenizer # Fast
from tqdm import tqdm

from shared.constants.constants import *
from applications.aiEngine.utilities import CustomDataset, CustomNonPaddingTokenLoss
from torch.utils.data import Dataset, DataLoader


class Trainer:
    def __init__(self, data_frame, jmdl):
        self.data_frame = data_frame
        self.model = jmdl
        self.optimizer = AdamW(jmdl.parameters(), lr=3e-5)
        self.pre_trained_model = True
        self.num_epochs = NUM_EPOCHS

        self.optimizer = jmdl.optimizer
        self.loss = jmdl.loss
        self.train_ratio = .85
        self.train_size = int(len(data_frame) * self.train_ratio)
        self.valid_size = len(data_frame) - self.train_size

    def train(self):

        train_dataset, valid_dataset = torch.utils.data.random_split(self.data_frame, [self.train_size, self.valid_size])

        # Initialize PyTorch Dataset
        train_dataset = CustomDataset(train_dataset, self.model.robertaTokenizer, self.model.slot2idx, self.model.intent2idx)
        valid_dataset = CustomDataset(valid_dataset, self.model.robertaTokenizer, self.model.slot2idx, self.model.intent2idx)

        # Create PyTorch DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        slot_loss_fn = self.loss['slots']
        intent_loss_fn =  self.loss['intent']


        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                # Move inputs and labels to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_label'].to(device)
                slot_labels = batch['slot_labels'].to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                slot_logits, intent_logits = self.model(input_ids, attention_mask)

                # Calculate losses
                slot_loss = slot_loss_fn(slot_logits, slot_labels)
                intent_loss = intent_loss_fn(intent_logits, intent_labels)

                # Total loss (combine slot and intent losses)
                total_loss = slot_loss + .8 * intent_loss

                # Backward pass and optimization step
                total_loss.backward()
                self.optimizer.step()

            # Validation steps...
            self.model.eval()  # Set the model to evaluation mode
            val_slot_loss, val_intent_loss, val_slot_accuracy, val_intent_accuracy = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch in tqdm(valid_dataloader, desc=f"Validation - Epoch {epoch + 1}/{self.num_epochs}"):
                    # Move inputs and labels to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    intent_labels = batch['intent_label'].to(device)
                    slot_labels = batch['slot_labels'].to(device)

                    # Forward pass
                    slot_logits, intent_logits = self.model(input_ids, attention_mask)

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

                print(f"Average Validation Slot Loss: {val_slot_loss}")
                print(f"Average Validation Intent Loss: {val_intent_loss}")
                print(f"Average Validation Slot Accuracy: {val_slot_accuracy}")
                print(f"Average Validation Intent Accuracy: {val_intent_accuracy}")


                # Callback to check if it's the best epoch and save the model
                self.model.combine_callback.on_epoch_end(epoch, val_slot_loss, val_intent_loss, val_slot_accuracy,
                                                 val_intent_accuracy)

