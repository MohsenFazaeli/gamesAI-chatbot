from shared.constants.a4_qgen import *

# AI CONSTATNTS
MAX_LENTGH = 30
# Train the model using fit_generator
BATCH_SIZE = 128
NUM_EPOCHS = 3

NUM_BATCHES = int(number_of_questions / BATCH_SIZE) + 1  # Adjust based on your dataset size
CHECKPOINT_DIR = ".storage/chechpoints/"
# model_name = "bert-base-uncased"
# model_name = "xlm-roberta-base"
# model_name = "HooshvareLab/bert-fa-zwnj-base"
#
# model_name = "../models/xlm-roberta-large"
model_name = "xlm-roberta-large"
hf_path = ".storage/hf/xlmr"
conf_path = ".storage/xlmr/conf"
base_path = ".storage/xlmr/base"
# base_path = '.storage/XLMRoberta/brt.pth'
tokenizer_path = ".storage/xlmr/tokenizer"
xlm_mixed_path = ".storage/xlmr/mixed"
final_path = ".storage/xlmr/final"


# #TODO needs more variations for each question
# #model_name = "bert-base-uncased"
# model_name = "xlm-roberta-base"
# #model_name = "HooshvareLab/bert-fa-zwnj-base"
#
# tokenizer = BertTokenizer.from_pretrained(model_name)
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
