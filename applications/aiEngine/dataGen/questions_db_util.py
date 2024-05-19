import random
import pandas as pd
from shared.constants.constants import *
from applications.db.db_utils import *
class QDB():
    # E2P_map = {'1': '۱', '2': '۲', '3': '۳', '4': '۴', '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹', '0': '۰'}
    #
    # # تبدیل اعداد انگلیسی به فارسی در رشته ورودی
    # def convert_number_to_persian(strIn: str):
    #     a = map(lambda ch: self.E2P_map[ch] if ch in E2P_map else ch, strIn)
    #     return ''.join(list(a))

    def __init__(self):
        self.random = random.SystemRandom()
        # self.cities = pd.read_csv(r"app\\resources\\irancities.csv")
        self.cities = cities
        self.first_names = first_names['farsi_names'].values.tolist()
        self.last_names = last_names['last_name'].values.tolist()

        # TODO un comment following line to get persian barracks name
        self.barracks_name = self.cities['city_FA'].values.tolist()

        self.sensors = get_all_sensors()
        # TODO un comment following line to get persian sensor name
        #  self.sensors_name = list(self.sensors['name_FA'])

        self.barracks = get_all_baracks()
        self.sttaffs = get_all_staffs()

        self.SENSORS_TYPES_FA = []
        for i, v in SENSORS_TYPE_FA.items():
            self.SENSORS_TYPES_FA.extend(v)

        self.STATUS_FA = []
        for i, v in STATUS_FA.items():
            self.STATUS_FA.extend(v)

        self.MILITARY_RANKS_FA = []
        for i, v in MILITARY_RANKS_FA.items():
            self.MILITARY_RANKS_FA.extend(v)

        self.DIRECTIONS_FA = []
        for i, v in DIRECTIONS_FA.items():
            self.DIRECTIONS_FA.extend(v)

    def gen_barrcks_and_sensor(self):
        sensor = self.random.choice(self.SENSORS_TYPES_FA)
        barracks = self.random.choice(self.barracks_name)
        return {'sensor_type_1': sensor, 'barracks_name_1': barracks}

    def count_barracks_sensor_based_on_sensor_type(self):
        return self.gen_barrcks_and_sensor()

    def get_sensor_IP(self):
        sensor = self.random.choice(self.SENSORS_TYPES_FA)
        sensor_name = str(self.random.choice(list(self.sensors['ID'])))
        return {'sensor_type_1': sensor, 'sensor_name_1': sensor_name}

    def get_coordinates_of_sensor_1(self):
        return self.get_sensor_IP()

    def get_coordinates_of_sensor_2(self):
        return self.get_sensor_IP()

    def get_all_parameters_of_sensor(self):
        return self.get_sensor_IP()

    def count_personnel_in_barracks(self):
        barracks = self.random.choice(self.barracks_name)
        return {'barracks_name_1': barracks}

    def count_entities_of_type(self):
        sensor_type = self.random.choice(self.SENSORS_TYPES_FA)
        sensor_type2 = self.random.choice(self.SENSORS_TYPES_FA)
        return {'entity_type_1': sensor_type, 'specific_type_1': sensor_type2}

    def count_personnel_with_status_rank_in_barracks(self):
        status = self.random.choice(self.STATUS_FA)
        rank = self.random.choice(self.MILITARY_RANKS_FA)
        barracks = self.random.choice(self.barracks_name)
        return {"status_1": status, "rank_1": rank, "barracks_name_1": barracks}

    def barracks_connection_type(self):
        barracks_1 = self.random.choice(self.barracks_name)
        barracks_2 = self.random.choice(self.barracks_name)

        return {"barracks_name_1": barracks_1,
                "barracks_name_2": barracks_2}

    def count_sensors_of_type_in_barracks(self):
        sensor_type = self.random.choice(self.SENSORS_TYPES_FA)
        barracks_1 = self.random.choice(self.barracks_name)
        return {
            "sensor_type_1": sensor_type,
            "barracks_name_1": barracks_1}

    def sensor_barracks_connection_type(self):
        res = {
            "sensor_type_1": "{sensor_type_1}",
            "sensor_name_1": "{sensor_name_1}",
            "barracks_name_1": "{barracks_name_1}"
        }
        res.update(self.get_sensor_IP())
        res['barracks_name_1'] = self.random.choice(self.barracks_name)
        return res

    def sensor_frequency(self):
        return self.get_sensor_IP()

    def officer_serving_barracks(self):
        rank_1 = self.random.choice(self.MILITARY_RANKS_FA)
        name = self.random.choice(self.first_names)
        surname = self.random.choice(self.last_names)
        full_name = surname if self.random.random() > 0.5 else name + " " + surname
        return {"rank_1": rank_1,
                "officer_name_1": full_name}

    def radar_status_in_area(self):
        sensor_type = self.random.choice(self.SENSORS_TYPES_FA)
        direction = self.random.choice(self.DIRECTIONS_FA)
        return {
            "sensor_type_1": sensor_type,
            "area_name_1": direction,
        }

    def status_of_entities_of_type(self):

        return self.count_entities_of_type()

    def count_personnel_with_rank_in_barracks(self):
        rank_1 = self.random.choice(self.MILITARY_RANKS_FA)
        barracks_1 = self.random.choice(self.barracks_name)
        return {"rank_1": rank_1,
                "barracks_name_1": barracks_1}

    def count_personnel_with_rank_in_region(self):
        rank_1 = self.random.choice(self.MILITARY_RANKS_FA)
        direction = self.random.choice(self.DIRECTIONS_FA)
        return {"rank_1": rank_1, "region_1": direction}

    def status_of_communication_between_sensor_types_in_region(self):
        sensor_type_1 = self.random.choice(self.SENSORS_TYPES_FA)
        sensor_type_2 = self.random.choice(self.SENSORS_TYPES_FA)
        direction = self.random.choice(self.DIRECTIONS_FA)
        return {"sensor_type_1": sensor_type_1, "sensor_type_2": sensor_type_2, "region_1": direction}

    def count_sensors_of_specific_type_in_barracks(self):
        specific_type_1 = self.random.choice(self.SENSORS_TYPES_FA)
        barracks_name_1 = self.random.choice(self.barracks_name)
        return {"specific_type_1": specific_type_1, "barracks_name_1": barracks_name_1}

    def casual_greetings(self):
        return {}

    def casual_goodbyes(self):
        return {}
if __name__ == '__main__':
    qdb = QDB()

    df = qdb.gen_count_baraks_sensor_based_on_sensor_type()
    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)
