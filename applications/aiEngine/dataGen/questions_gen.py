import json
import os
import random
import copy

from shared.constants.constants import *
from shared.questions_template import *
from applications.aiEngine.dataGen.questions_db_util import QDB
from applications.utilites.text_utilites import remove_punctuation

class QUESTION_GEN:
    def __init__(self, qdb):
        self.random = random.SystemRandom()
        self.questions_db = qdb

    def generate_question(self, question_dict, entity_values):
        """
        Generate a question by replacing placeholders with entity values.

        Args:
        - question_dict (dict): The question dictionary.
        - entity_values (dict): A dictionary containing values for named entities.

        Returns:
        - str: The generated question.
        """
        template = self.random.choice(question_dict["TEMPLATE"])
        new_template = (template + '.')[:-1]

        for key, value in entity_values.items():
            new_template = new_template.replace(f"{{{key}}}", "_")

        word_tokenized_template = new_template.split()

        ner_tags = ["_" if t == "_" else 'O' for t in word_tokenized_template]
        # for key, value in ner_vals.items():
        #     NER_TAGS = NER_TAGS.replace("_", "B-"+key)

        # Replace placeholders with entity values
        for key, value in entity_values.items():
            if key in template:
                ner_value = value.split()
                idx = ner_tags.index('_')
                ner_tags[idx] = "B-" + key

                word_tokenized_template[idx] = ner_value[0]
                # Check if the placeholder spans multiple words

                if len(ner_value) > 1:
                    for i, sub_token in enumerate(ner_value[1:]):
                        ner_tags.insert(i + idx + 1, f"I-{key}")
                        word_tokenized_template.insert(i + idx + 1, sub_token)

        return ner_tags, word_tokenized_template


    def from_json_to_csv(self, json_file):
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

    def generate(self):
        # q_gen = QUESTION_GEN(tokenizer)
        q_gen = self
        qdb = self.questions_db
        file_number = 0
        questions = {}
        q_temps_keys = [v for v in question_temps]  # ['sensor_barracks_connection_type'] # [v for v in question_temps]
        n_questions = len(q_temps_keys)
        for i in range(number_of_questions):

            # for temp_key in q_temps_keys:
            temp_key = q_temps_keys[i%n_questions]
            temp = question_temps[temp_key]
            f = getattr(qdb, temp_key)
            data = f()
            q = { "INTENTS": temp["INTENTS"]}

            new_tags, question_text = q_gen.generate_question(temp, data)

            # aligned_tags = q_gen.align_tags_with_tokens(new_question, new_tags)
            q["NER_TAGS"] = new_tags
            q['TEXT'] = question_text

            bad_tags = [i for i in new_tags if not (i.endswith('_1') or i.endswith("_2") or i == 'O')]

            if len(bad_tags) > 0:
                print("Bad tags: ", bad_tags)
                print(q)
                exit()

            questions[len(questions)] = q
            print(q['TEXT'], q['NER_TAGS'])
            print(len(q['TEXT']), len(q["NER_TAGS"]))
            if len(q['TEXT']) != len(q["NER_TAGS"]):
                print(temp_key)
                for i,j in zip(q["TEXT"], q["NER_TAGS"]):
                    print(i, '==>', j)
                exit()

        os.makedirs(questions_dirctory, exist_ok=True )
        with open(questions_dirctory + 'q_roberta_' + str(file_number) + '.json', 'w', encoding='utf-8') as fp:
            # json.dump(questions, fp,ensure_ascii=False)
            json.dump(questions, fp, ensure_ascii=False, indent=4)

        self.df = self.from_json_to_csv(questions_dirctory + 'q_roberta_0.json')

        print(questions_dirctory + 'q_roberta_' + str(file_number) + '.json')
        return self.df

if __name__ == '__main__':
    q_gen = QUESTION_GEN(QDB())
    q_gen.generate()
