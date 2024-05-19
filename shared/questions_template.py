import json
from shared.constants.a4_qgen import questions_template_json_path

question_temps_depricated = {
    "count_barracks_sensor_based_on_sensor_type": {
        "TEMPLATE": [
            "چند سنسور از نوع "
            + "{sensor_type_1}"
            + " مربوط به پادگان "
            + "{barracks_name_1}"
            + " وجود دارد؟"],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "O",
            "B-sensor_type_1",
            "O",
            "O",
            "O",
            "B-barracks_name_1",
            "O",
            "O",
        ],
        "NERVALS": {
            "sensor_type_1": "{sensor_type_1}",
            "barracks_name_1": "{barracks_name_1}",
        },
        "INTENTS": ["count_barracks_sensor_based_on_sensor_type"],
    },
    "get_sensor_IP": {
        "TEMPLATE": ["آیپی "
                     + "{sensor_type_1}"
                     + " "
                     + "{sensor_name_1}"
                     + " چیست؟"],
        "NERTAGS": ["O", "B-sensor_type_1", "B-sensor_name_1", "O"],
        "NERVALS": {
            "sensor_type_1": "{sensor_type_1}",
            "barracks_name_1": "{barracks_name_1}",
        },
        "INTENTS": ["get_sensor_IP"],
    },
    "get_coordinates_of_sensor_1": {
        "TEMPLATE": ["مختصات جغرافیایی "
                     + "{sensor_type_1}"
                     + " "
                     + "{sensor_name_1}"
                     + " چیست؟"],
        "NERTAGS": ["O", "O", "B-sensor_type_1", "B-sensor_name_1", "O"],
        "NERVALS": {
            "sensor_name_1": "{sensor_name_1}",
            "sensor_type_1": "{sensor_type_1}",
        },
        "INTENTS": ["get_coordinates_of_sensor"],
    },
    "get_coordinates_of_sensor_2": {
        "TEMPLATE": ["مختصات "
                     + "{sensor_type_1}"
                     + " "
                     + "{sensor_name_1}"
                     + " چیست؟"],
        "NERTAGS": ["O", "B-sensor_type_1", "B-sensor_name_1", "O"],
        "NERVALS": {
            "sensor_name_1": "{sensor_name_1}",
            "sensor_type_1": "{sensor_type_1}",
        },
        "INTENTS": ["get_coordinates_of_sensor"],
    },
    "get_all_parameters_of_sensor": {
        "TEMPLATE": ["پارامترهای مربوط به "
                     + "{sensor_type_1}"
                     + " "
                     + "{sensor_name_1}"
                     + " دارای چه مقادیری هستند؟"],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "B-sensor_type_1",
            "B-sensor_name_1",
            "O",
            "O",
            "O",
            "O",
        ],
        "NERVALS": {
            "sensor_name_1": "{sensor_name_1}",
            "sensor_type_1": "{sensor_type_1}",
        },
        "INTENTS": ["get_all_parameters_of_sensor"],
    },
    "count_personnel_in_barracks": {
        "TEMPLATE": [
            "چه تعداد پرسنل فعال در پادگان "
            + "{barracks_name_1}"
            + " وجود دارد؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-barracks_name_1",
            "O",
            "O"
        ],
        "NERVALS": {
            "barracks_name_1": "{barracks_name_1}"
        },
        "INTENTS": ["count_personnel_in_barracks"]
    },
    "count_personnel_with_status_rank_in_barracks": {
        "TEMPLATE": [
            "چه تعداد پرسنل {status_1} با {rank_1} در پادگان "
            + "{barracks_name_1}"
            + " وجود دارد؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "B-status_1",
            "O",
            "B-rank_1",
            "O",
            "O",
            "B-barracks_name_1",
            "O",
            "O"
        ],
        "NERVALS": {
            "status_1": "{status_1}",
            "rank_1": "{rank_1}",
            "barracks_name_1": "{barracks_name_1}"
        },
        "INTENTS": ["count_personnel_with_status_rank_in_barracks"]
    },
    "barracks_connection_type": {
        "TEMPLATE": [
            "نوع ارتباط "
            + "پادگان "
            + "{barracks_name_1}"
            + " و "
            + "{barracks_name_2}"
            + " به چه صورت است؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "B-barracks_name_1",
            "O",
            "B-barracks_name_2",
            "O",
            "O",
            "O",
            "O"
        ],
        "NERVALS": {
            "barracks_name_1": "{barracks_name_1}",
            "barracks_name_2": "{barracks_name_2}"
        },
        "INTENTS": ["barracks_connection_type"]
    },
    "count_sensors_of_type_in_barracks": {
        "TEMPLATE": [
            "چه تعداد سنسور از نوع "
            + "{sensor_type_1}"
            + " در پادگان "
            + "{barracks_name_1}"
            + " وجود دارد؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "O",
             "O",
            "B-sensor_type_1",
            "O",
            "O",
            "B-barracks_name_1",
            "O",
            "O",
        ],
        "NERVALS": {
            "sensor_type_1": "{sensor_type_1}",
            "barracks_name_1": "{barracks_name_1}"
        },
        "INTENTS": ["count_sensors_of_type_in_barracks"]
    },
    "sensor_barracks_connection_type": {
        "TEMPLATE": [
            "نوع ارتباط "
            + "{sensor_type_1} "
            + "{sensor_name_1}"
            + " و پادگان "
            + "{barracks_name_1}"
            + " به چه صورت است؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "B-sensor_type_1",
            "B-sensor_name_1",
            "O",
            "O",
            "O",
            "B-barracks_name_1",
            "O",
            "O",
            "O"
        ],
        "NERVALS": {
            "sensor_type_1": "{sensor_type_1}",
            "sensor_name_1": "{sensor_name_1}",
            "barracks_name_1": "{barracks_name_1}"
        },
        "INTENTS": ["sensor_barracks_connection_type"]
    },
    "sensor_frequency": {
        "TEMPLATE": [
            " {sensor_type_1} "
            + "{sensor_name_1}"
            + " با چه فرکانس کاری کار می‌کند؟"
        ],
        "NERTAGS": [
            "B-sensor_type_1",
            "B-sensor_name_1",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
        ],
        "NERVALS": {
            "sensor_type_1": "{sensor_type_1}",
            "sensor_name_1": "{sensor_name_1}"
        },
        "INTENTS": ["sensor_frequency"]
    },
    "officer_serving_barracks": {
        "TEMPLATE": [
            "{rank_1} "
            + "{officer_name_1}"
            + " در کدام پادگان خدمت می‌کند؟"
        ],
        "NERTAGS": [
            "B-rank_1",
            "B-officer_name_1",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        "NERVALS": {
            "rank_1": "{rank_1}",
            "officer_name_1": "{officer_name_1}"
        },
        "INTENTS": ["officer_serving_barracks"]
    }, "radar_status_in_area": {
        "TEMPLATE": [
            "وضعیت " +
            "{sensor_type_1}" +
            " محدوده "
            + "{area_name_1}"
            + " چگونه است؟"
        ],
        "NERTAGS": [
            "O",
            "B-sensor_type_1",
            "O",
            "B-area_name_1",
            "O",
            "O"
        ],
        "NERVALS": {
            "sensor_type_1": "{sensor_type_1}",
            "area_name_1": "{area_name_1}",
        },
        "INTENTS": ["sensor_status_in_area"]
    },
    "count_entities_of_type": {
        "TEMPLATE": [
            "کلا چند {entity_type_1} از نوع "
            + " {specific_type_1} "
            + " وجود دارد؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "O",
            "B-entity_type_1",
            "O",
            "B-specific_type_1",
            "O",
            "O"
        ],
        "NERVALS": {
            "entity_type_1": "{entity_type_1}",
            "specific_type_1": "{specific_type_1}"
        },
        "INTENTS": ["count_entities_of_type"]
    },
    "status_of_entities_of_type": {
        "TEMPLATE": [
            "وضعیت "
            " {entity_type_1} "
            " نوع "
            + "{specific_type_1}"
            + " چیست؟"
        ],
        "NERTAGS": [
            "O",
            "B-entity_type_1",
            "O",
            "B-specific_type_1",
            "O",
        ],
        "NERVALS": {
            "entity_type_1": "{entity_type_1}",
            "specific_type_1": "{specific_type_1}"
        },
        "INTENTS": ["status_of_entities_of_type"]
    },
    "count_personnel_with_rank_in_barracks": {
        "TEMPLATE": [
            "چند نفر پرسنل با درجه "
            + "{rank_1}"
            + " در پادگان "
            + "{barracks_name_1}"
            + " وجود دارد؟"
        ],
        "NERTAGS": [
            "O",
            "O",
            "O",
             "O",
            "O",
            "B-rank_1",
            "O",
             "O",
            "B-barracks_name_1",
            "O",
            "O"
        ],
        "NERVALS": {
            "rank_1": "{rank_1}",
            "barracks_name_1": "{barracks_name_1}"
        },
        "INTENTS": ["count_personnel_with_rank_in_barracks"]
    },
    "count_personnel_with_rank_in_region": {
        "TEMPLATE": ["چند نفر از پرسنل با درجه {rank_1} در ناحیه "
                     "{region_1}"
                     " وجود دارند؟"],
        "NERTAGS": ["O", "O", "O","O","O","O", "B-rank_1", "O","O", "B-region_1", "O", "O"],
        "NERVALS": {"rank_1": "{rank_1}", "region_1": "{region_1}"},
        "INTENTS": ["count_personnel_with_rank_in_region"]
    }
    ,
    "status_of_communication_between_sensor_types_in_region": {
        "TEMPLATE": ["نوع ارتباط سنسورهای {sensor_type_1} و {sensor_type_2} در منطقه {region_1} چگونه است؟"],
        "NERTAGS": ["O", "O", "O", "B-sensor_type_1", "O", "B-sensor_type_2", "O", "O","B-region_1", "O", "O"],
        "NERVALS": {"sensor_type_1": "{sensor_type_1}", "sensor_type_2": "{sensor_type_2}", "region_1": "{region_1}"},
        "INTENTS": ["status_of_communication_between_sensor_types_in_region"]
    }
    , "count_sensors_of_specific_type_in_barracks": {
        "TEMPLATE": ["چند رادار از نوع {specific_type_1} در پادگان {barracks_name_1} وجود دارد؟"],
        "NERTAGS": ["O", "O", "O", "O", "B-specific_type_1", "O", "O", "B-barracks_name_1", "O", "O"],
        "NERVALS": {"specific_type_1": "{specific_type_1}", "barracks_name_1": "{barracks_name_1}"},
        "INTENTS": ["count_sensors_of_specific_type_in_barracks"]
    }
    , "casual_greetings": {
        "TEMPLATE": ["سلام!", "سلام دوست عزیز!", "هر کاری داریم می‌تونیم کمک کنیم."],
        "NERTAGS": ["O"],
        "INTENTS": ["casual_greetings"]
    }
    , "casual_goodbyes": {
        "TEMPLATE": ["خداحافظ!", "خدا نگهدار!", "به امید دیدار!", "هر وقت نیاز داشتی اینجا هستیم."],
        "NERTAGS": ["O"],
        "INTENTS": ["casual_goodbyes"]
    }
}

question_temps = question_temps_depricated
try:
    # Load the JSON data from the file into a Python dictionary
    with open(questions_template_json_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)
    question_temps = data_dict
except:
    pass


if __name__ == "__main__":
    with open(questions_template_json_path, 'w',encoding='utf-8') as fp:
        json.dump(question_temps_depricated, fp, ensure_ascii=False, indent=4)

    from applications.utilites.text_utilites import *
    from hazm import Normalizer
    normalizer = Normalizer()
    for k, v in question_temps_depricated.items():
        text_template = (v['TEMPLATE'][0]).strip().replace("\u200c","")
        text_template = remove_punctuation(text_template)
        text_tokens = text_template.split()

        if len(text_tokens) != len(v["NERTAGS"]):
            for tmp in zip(text_tokens, v["NERTAGS"]):
                print(tmp)

            print(v)
            print(k)
            exit(1)


        for tmp in zip(text_tokens, v["NERTAGS"]):
            print(tmp[1],'=>',tmp[0])

        print()
        print()
        print()






