import torch
from shared.constants.constants import *
from applications.db.api_provider import *

from applications.db import a1_build_db
from applications.frontend import a2_synthesizeMap, a3_Upgrade_map_manual


from applications.aiEngine.dataGen.questions_gen import QUESTION_GEN
from applications.aiEngine.dataGen.questions_db_util import QDB

q_gen = QUESTION_GEN(QDB())
q_gen.generate()

from applications.aiEngine.train.Trainer import Trainer
from applications.aiEngine.test.Tester import Tester
from applications.aiEngine.model.JointIntentAndSlotFillingModel import JointIntentAndSlotFillingModel as JM
from transformers import AutoModel as OurModel, AutoTokenizer as OurTokenizer
from shared.constants.constants import model_name, base_path, hf_path, xlm_mixed_path

xlmr = None

try:
    xlmr = OurModel.from_pretrained(xlm_mixed_path)
    print("Base Model loaded as it should")
except Exception as e:
    print(f"Base path {xlm_mixed_path} is not valid", e)
#print(xlmr)

if xlmr is None or True:
    try:
        xlmr = OurModel.from_pretrained(hf_path)
    except Exception as e:
        print(f"Base path {hf_path} is not valid", e)
        xlmr = OurModel.from_pretrained(model_name)
    #xlmr.save_pretrained()
else:
    pass
    # print("Model loaded as it should")

model = JM(bert=xlmr, method='Thick')
model.load_and_save_pretrained_model()
print("Model is Pretrained: ", model.pre_trained_model)

# target_layer_name = "bert"M
# target_parameters = []
# for name, param in model.named_parameters():
#     #if target_layer_name not in name or True:
#     print(name, param.shape)
#     target_parameters.append(param)

# Save the model
# import torch
# torch.save(model.state_dict(), 'sample_model.pth')

# Load the model
# loaded_model = JM(bert=xlmr, method='light')
# loaded_model.load_state_dict(torch.load('sample_model.pth'), strict=False)

# Access parameters of the loaded model
# for name, param in loaded_model.named_parameters():
#     print(f"Layer: {name}, Size: {param.size()}")


if not model.pre_trained_model:
    trainer = Trainer(q_gen.df, model)
    trainer.train()

    tester = Tester(q_gen.df, model)
    tester.test()

#few single tests:
tester = Tester(q_gen.df, model)
api = QueryAPI()
for idx,sample in q_gen.df.iloc[:15].iterrows():
    processed = tester.nlu(sample.sentence)
    print(f"{idx} Question Analysis:")
    print(sample.sentence)
    print(processed)
    print(api.query(processed))


