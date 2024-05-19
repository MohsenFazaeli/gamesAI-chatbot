import random
import pandas as pd
from shared.constants.constants import *
from shared.answers_template import answer_templates
from applications.db.db_utils import *


class QueryAPI():

    def __init__(self):
        self.debug = True
        self.conn = create_connection(DB_NAME)
        self.random = random.SystemRandom()
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

    def query(self, processed):
        slots = processed[0]
        intent = processed[-1]
        if hasattr(self, intent):
            f = getattr(self, intent)
            out = f(slots, intent)
            return out

    def count_barracks_sensor_based_on_sensor_type(self, params, intent):
        br_name = params['barracks_name_1']
        sensor_type = params['sensor_type_1']

        sensor_type_en, sensor_type_fa = "", ""
        for sensor_type_en, sensor_type_fa in SENSORS_TYPE_FA.items():
            if sensor_type in sensor_type_fa:
                break
        else:
            pass  # Todo error handeling


        # Ensure the connection to the database is established first
        # Assuming 'conn' is a valid database connection object

        # Using placeholders in the SQL query to prevent SQL injection
        # Also, assuming 'conn' supports parameterized queries
        br_id_cursor = self.conn.execute(
            """SELECT ID FROM barracks
            WHERE name = ? or name_fa = ?""",
            (br_name, br_name)  # Pass the parameters separately to avoid SQL injection
        )
        # Fetch the result from the cursor
        br_id_result = br_id_cursor.fetchone()

        # Check if the result is not empty before accessing its value
        if br_id_result:
            br_id = br_id_result[0]  # Extracting the ID from the result tuple
            if self.debug:
                print("پادگان پیدا شد", br_id )
        else:
            # Handle the case where no barracks with the given name is found
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["baranks_not_found"].format(barracks_name_1=br_name,
                                                                               sensor_type_1=sensor_type)
            return template
        if sensor_type_en in ["None","Sensor"]:
            counted_br_cursor = self.conn.execute(
            """
            SELECT count(*) FROM sensors
            WHERE barracks_ID = ? """,
            (br_id,)  # Pass the parameters separately to avoid SQL injection
        )
        else:
            # Using placeholders and parameters for the sensor type and barracks ID
            counted_br_cursor = self.conn.execute(
            """
            SELECT count(*) FROM sensors
            WHERE barracks_ID = ? and sensor_type = ?""",
            (br_id, sensor_type_en)  # Pass the parameters separately to avoid SQL injection
        )

        # Fetch the result from the cursor
        counted_br_result = counted_br_cursor.fetchone()

        # Check if the result is not empty before returning it
        if counted_br_result:
            counted_br = counted_br_result[0]  # Extracting the count from the result tuple
            if counted_br == 0:
                template = answer_templates[intent]["no_sensor_4_barracks"].format(barracks_name_1=br_name,
                                                                                   sensor_type_1=sensor_type)

                if self.debug: print("تعداد سنسور شمرده شده صفر", sensor_type_en )

                return template

            template = answer_templates[intent]["final"].format(barracks_name_1=br_name, sensor_type_1=sensor_type,
                                                                scounted_snsr=counted_br)
            return template
        else:
            # Handle the case where no sensors are found for the given barracks ID and sensor type
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["no_sensor_4_barracks"].format(barracks_name_1=br_name,
                                                                               sensor_type_1=sensor_type_fa)
            if self.debug: print("جواب شمارش سنسور مشکل دارد")

            return template

        if self.debug: print("End of count_barracks_sensor_based_on_sensor_type")


    def get_sensor_IP(self, params, intent):
        sensor_name = params['sensor_name_1']
        sensor_type = params['sensor_type_1']
        # Ensure the connection to the database is established first
        # Assuming 'conn' is a valid database connection object

        # Using placeholders in the SQL query to prevent SQL injection
        # Also, assuming 'conn' supports parameterized queries
        sensor_type_en, sensor_type_fa = "", ""
        for sensor_type_en, sensor_type_fa in SENSORS_TYPE_FA.items():
            if sensor_type_fa == sensor_type:
                break
        else:
            pass  # Todo error handeling

        sensor_ip_cursor = self.conn.execute(
            """SELECT IP FROM sensors
            WHERE ID = ? and sensor_type = ?""",  # TODO change ID to names
            (sensor_name, sensor_type_en)  # Pass the parameters separately to avoid SQL injection
        )
        sensor_ip_result = sensor_ip_cursor.fetchone()

        if sensor_type_en == "None" or sensor_type_en == "Sensor" or not sensor_ip_result:
            sensor_ip_cursor = self.conn.execute(
                """SELECT IP FROM sensors
                WHERE ID = ?""",  # TODO change ID to names
                (sensor_name,)  # Pass the parameters separately to avoid SQL injection
            )

        # Fetch the result from the cursor
        sensor_ip_result = sensor_ip_cursor.fetchone()

        # Check if the result is not empty before returning it
        if sensor_ip_result:
            sensor_ip = sensor_ip_result[0]  # Extracting the IP address from the result tuple
            template = answer_templates[intent]["Final"].format(IP=sensor_ip)
            return template
        else:
            # Handle the case where no sensor with the given name and type is found
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["sensor_not_found"]
            return template

    def get_coordinates_of_sensor(self, params, intent):
        if "sensor_name_1" not in params:
            return "نام سنسور به درستی شناسایی نشد"

        sensor_name = params['sensor_name_1']
        sensor_type = params['sensor_type_1']
        # Ensure the connection to the database is established first
        # Assuming 'conn' is a valid database connection object

        # Using placeholders in the SQL query to prevent SQL injection
        # Also, assuming 'conn' supports parameterized queries
        sensor_type_en, sensor_type_fa = "", ""
        for sensor_type_en, sensor_type_fa in SENSORS_TYPE_FA.items():
            if sensor_type_en == sensor_type:
                break
        else:
            pass  # Todo error handeling

        sensor_cord_cursor = self.conn.execute(
            """SELECT longitude,latitude FROM sensors
            WHERE ID = ? and sensor_type = ?""",  # TODO change ID to names
            (sensor_name, sensor_type_en)  # Pass the parameters separately to avoid SQL injection
        )

        sensor_cord_result = sensor_cord_cursor.fetchone()

        if sensor_type_en == "None" or sensor_type_en == "Sensor" or not sensor_cord_cursor:
            sensor_cord_cursor = self.conn.execute(
                """SELECT longitude,latitude FROM sensors
                WHERE ID = ?""",  # TODO change ID to names
                (sensor_name,)  # Pass the parameters separately to avoid SQL injection
            )

        # Fetch the result from the cursor
        sensor_cord_result = sensor_cord_cursor.fetchone()

        # Check if the result is not empty before returning it
        if sensor_cord_result:
            long,lat = sensor_cord_result  # Extracting the IP address from the result tuple
            template = answer_templates[intent]["Final"].format(lat=lat, long=long)
            return template
        else:
            # Handle the case where no sensor with the given name and type is found
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["sensor_not_found"]
            return template


    def get_all_parameters_of_sensor(self, params, intent):
        sensor_name = params['sensor_name_1']
        sensor_type = params['sensor_type_1']
        # Ensure the connection to the database is established first
        # Assuming 'conn' is a valid database connection object

        # Using placeholders in the SQL query to prevent SQL injection
        # Also, assuming 'conn' supports parameterized queries
        sensor_type_en, sensor_type_fa = "", ""
        for sensor_type_en, sensor_type_fa in SENSORS_TYPE_FA.items():
            if sensor_type_en == sensor_type:
                break
        else:
            pass  # Todo error handeling

        sensor_cord_cursor = self.conn.execute(
            """SELECT parameters FROM sensors
            WHERE ID = ? and sensor_type = ?""",  # TODO change ID to names
            (sensor_name, sensor_type_en)  # Pass the parameters separately to avoid SQL injection
        )

        sensor_cord_result = sensor_cord_cursor.fetchone()

        if sensor_type_en == "None" or not sensor_cord_result :
            sensor_cord_cursor = self.conn.execute(
                """SELECT parameters FROM sensors
                WHERE ID = ?""",  # TODO change ID to names
                (sensor_name,)  # Pass the parameters separately to avoid SQL injection
            )

        # Fetch the result from the cursor
        sensor_cord_result = sensor_cord_cursor.fetchone()

        # Check if the result is not empty before returning it
        if sensor_cord_result:
            parameters = json.loads(sensor_cord_result[0])
            # Extracting the IP address from the result tuple
            formatted_parameters = "\n"+"\n".join([f"{key}: {value}" for key, value in parameters.items()])
            template = answer_templates[intent]["Final"].format(params = formatted_parameters)
            return template
        else:
            # Handle the case where no sensor with the given name and type is found
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["sensor_not_found"]
            return template

    def count_personnel_in_barracks(self, params, intent):
        barracks_name = params['barracks_name_1']

        # Assuming 'conn' is a valid database connection object
        cursor = self.conn.cursor()

        try:
            # Fetch barracks_ID from the barracks table based on barracks_name
            cursor.execute(
                "SELECT ID FROM barracks WHERE name_fa = ? or name =?",
                (barracks_name,barracks_name,)
            )
            barracks_id = cursor.fetchone()

            if barracks_id:  # If barracks ID is found
                barracks_id = barracks_id[0]  # Extract the ID from the result

                # Query the personnel table to count personnel in the specified barracks
                cursor.execute(
                    "SELECT COUNT(*) FROM staffs WHERE barracks_ID = ?",
                    (barracks_id,)
                )

                count = cursor.fetchone()[0]  # Get the count of personnel in the barracks

                if count > 0:
                    template = answer_templates[intent]["Final"].format(barracks_name_1=barracks_name, count=count)
                else:
                    template = answer_templates[intent]["personnel_not_found"].format(barracks_name_1=barracks_name)

                return template
            else:
                # Handle the case where no barracks with the given name is found
                template = answer_templates[intent]["barracks_not_found"].format(barracks_name_1=barracks_name)
                return template

        except Exception as e:
            print("Error:", e)
            # Handle the error, such as returning an error message or logging it


    def count_personnel_with_status_rank_in_barracks(self, params, intent):
        barracks_name = params['barracks_name_1']
        personnel_status = params['status_1']
        personnel_rank = params['rank_1']

        status_en, status_fa = "", ""
        for status_en, status_fa in STATUS_FA.items():
            if personnel_status in status_fa:
                break
        else:
            pass  # Todo error handeling

        rank_en, rank_fa = "", ""
        for rank_en, rank_fa in MILITARY_RANKS_FA.items():
            if personnel_rank in rank_fa:
                break
        else:
            pass  # Todo error handeling


        # Assuming 'conn' is a valid database connection object
        cursor = self.conn.cursor()

        try:
            # Fetch barracks_ID from the barracks table based on barracks_name
            cursor.execute(
                "SELECT ID FROM barracks WHERE name_fa = ? OR name = ?",
                (barracks_name, barracks_name,)
            )

            br_id_result = cursor.fetchone()

            # Check if the result is not empty before accessing its value
            if br_id_result:
                br_id = br_id_result[0]  # Extracting the ID from the result tuple
                if self.debug:
                    print("پادگان پیدا شد", br_id )
            else:
                # Handle the case where no barracks with the given name is found
                # You may raise an exception, return a specific value, or handle it differently based on your application logic
                template = answer_templates[intent]["baranks_not_found"].format(barracks_name_1=barracks_name)
                return template


            # Query the personnel table to count personnel with specific status and rank in the specified barracks
            if status_en in ["Active", "INACTIVE"]:
                cursor.execute(
                    "SELECT COUNT(*) FROM staffs WHERE barracks_ID = ? AND active = ? AND rank = ?",
                    (br_id, status_en=="ACTIVE", personnel_rank,)
                )
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM staffs WHERE barracks_ID = ? AND is_online = ? AND rank = ?",
                    (br_id, status_en=="ONLINE", personnel_rank,)
                )

            count = cursor.fetchone()[0]  # Get the count of personnel with the specified status and rank

            if count and count > 0:
                template = answer_templates[intent]["Final"].format(
                    barracks_name_1=barracks_name, count=count, status=personnel_status, rank=personnel_rank)
            else:
                template = answer_templates[intent]["personnel_not_found"].format(barracks_name_1=barracks_name)

            return template
        except Exception as e:
            print("Error:", e)


    def barracks_connection_type(self, params, intent):
        barracks_1 = params['barracks_name_1']
        barracks_2 = params['barracks_name_2']
        br_id_1 = br_id_2 = -1, -1
        cname_en, cname_fa = "",""


        # Assuming 'conn' is a valid database connection object
        cursor = self.conn.cursor()

        try:
            # Fetch barracks_ID from the barracks table based on barracks_name
            cursor.execute(
                "SELECT ID FROM barracks WHERE name_fa = ? OR name = ?",
                (barracks_1, barracks_1,)
            )

            br_id_result = cursor.fetchone()

            # Check if the result is not empty before accessing its value
            if br_id_result:
                br_id_1 = br_id_result[0]  # Extracting the ID from the result tuple
                if self.debug:
                    print("پادگان پیدا شد", br_id_1 )
            else:
                # Handle the case where no barracks with the given name is found
                # You may raise an exception, return a specific value, or handle it differently based on your application logic
                template = answer_templates[intent]["baranks_not_found"].format(barracks_name_1=barracks_1)
                return template


               # Fetch barracks_ID from the barracks table based on barracks_name
            cursor.execute(
                "SELECT ID FROM barracks WHERE name_fa = ? OR name = ?",
                (barracks_2, barracks_2,)
            )

            br_id_result = cursor.fetchone()

            # Check if the result is not empty before accessing its value
            if br_id_result:
                br_id_2 = br_id_result[0]  # Extracting the ID from the result tuple
                if self.debug:
                    print("پادگان پیدا شد", br_id_2 )
            else:
                # Handle the case where no barracks with the given name is found
                # You may raise an exception, return a specific value, or handle it differently based on your application logic
                template = answer_templates[intent]["baranks_not_found"].format(barracks_name_1=barracks_2)
                return template


            cursor.execute(
                "SELECT tye FROM links WHERE (ID1 = ? AND ID2 = ?) OR (ID1 = ? AND ID2 = ?)",
                (br_id_1, br_id_2,br_id_2,br_id_1)
            )

            connection_result = cursor.fetchone()

            if connection_result:

                for cname_en, cname_en in STATUS_FA.items():
                    if cname_en == connection_result:
                        cname_fa = cname_fa[0]
                        break
                else:
                    pass  # Todo error handeling

                template = answer_templates[intent]["final"].format(barracks_name_1=barracks_1, barracks_name_2=barracks_2,connection_type=cname_fa)

                return template
            else:
                template = answer_templates[intent]["connection_not_found"].format(barracks_name_1=barracks_1, barracks_name_2=barracks_2,connection_type=cname_fa)


        except Exception as e:
            print("Error:", e)

    def count_sensors_of_type_in_barracks(self, params, intent):
        return self.count_barracks_sensor_based_on_sensor_type(params, intent)
        barracks_name = params['barracks_name_1']
        sensor_type = params['sensor_type_1']

        sensor_type_en, sensor_type_fa = "",""
        for  sensor_type_en, sensor_type_fa  in SENSORS_TYPE_FA.items():
            if sensor_type in sensor_type_fa:
                break
        else:
            pass

        br_id_cursor = self.conn.execute(
            """SELECT ID FROM barracks
            WHERE name = ? or name_fa = ?""",
            (barracks_name, barracks_name)  # Pass the parameters separately to avoid SQL injection
        )
        # Fetch the result from the cursor
        br_id_result = br_id_cursor.fetchone()

        # Check if the result is not empty before accessing its value
        if br_id_result:
            br_id = br_id_result[0]  # Extracting the ID from the result tuple
            if self.debug:
                print("پادگان پیدا شد", br_id )
        else:
            # Handle the case where no barracks with the given name is found
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["baranks_not_found"].format(barracks_name_1=barracks_name,
                                                                               sensor_type_1=sensor_type)
            return template

        if sensor_type_en in ["None","Sensor"]:
            counted_br_cursor = self.conn.execute(
            """
            SELECT count(*) FROM sensors
            WHERE barracks_ID = ? """,
            (br_id,)  # Pass the parameters separately to avoid SQL injection
        )
        else:
            # Using placeholders and parameters for the sensor type and barracks ID
            counted_br_cursor = self.conn.execute(
            """
            SELECT count(*) FROM sensors
            WHERE barracks_ID = ? and sensor_type = ?""",
            (br_id, sensor_type_en)  # Pass the parameters separately to avoid SQL injection
        )

        # Fetch the result from the cursor
        counted_br_result = counted_br_cursor.fetchone()

        # Check if the result is not empty before returning it
        if counted_br_result:
            counted_br = counted_br_result[0]  # Extracting the count from the result tuple
            if counted_br == 0:
                template = answer_templates[intent]["no_sensor_4_barracks"].format(barracks_name_1=barracks_name,
                                                                                   sensor_type_1=sensor_type)

                if self.debug: print("تعداد سنسور شمرده شده صفر", sensor_type_en )

                return template

            template = answer_templates[intent]["final"].format(barracks_name_1=barracks_name, sensor_type_1=sensor_type,
                                                                scounted_snsr=counted_br)
            return template
        else:
            # Handle the case where no sensors are found for the given barracks ID and sensor type
            # You may raise an exception, return a specific value, or handle it differently based on your application logic
            template = answer_templates[intent]["no_sensor_4_barracks"].format(barracks_name_1=barracks_name,
                                                                               sensor_type_1=sensor_type_fa)
            if self.debug: print("جواب شمارش سنسور مشکل دارد")

            return template

        if self.debug: print("End of count_barracks_sensor_based_on_sensor_type")




    def count_entities_of_type(self):
        sensor_type = self.random.choice(self.SENSORS_TYPES_FA)
        sensor_type2 = self.random.choice(self.SENSORS_TYPES_FA)
        return {'entity_type_1': sensor_type, 'specific_type_1': sensor_type2}

    def sensor_barracks_connection_type(self, params, intent):
        barracks_name = params['barracks_name_1']
        sensor_name = params['sensor_name_1']
        sensor_type = params['sensor_type_1']
        br_id = snsr_id = -1, -1
        cname_en, cname_fa = "",""
        # Assuming 'conn' is a valid database connection object
        cursor = self.conn.cursor()

        try:
            # Fetch barracks_ID from the barracks table based on barracks_name
            cursor.execute(
                "SELECT ID FROM barracks WHERE name_fa = ? OR name = ?",
                (barracks_name, barracks_name,)
            )

            br_id_result = cursor.fetchone()

            # Check if the result is not empty before accessing its value
            if br_id_result:
                br_id = br_id_result[0]  # Extracting the ID from the result tuple
                if self.debug:
                    print("پادگان پیدا شد", br_id)
            else:
                # Handle the case where no barracks with the given name is found
                # You may raise an exception, return a specific value, or handle it differently based on your application logic
                template = answer_templates[intent]["baranks_not_found"].format(barracks_name_1=barracks_name)
                return template


            cursor.execute(
                "SELECT sensor_type FROM sensors WHERE barracks_ID=? AND ID=?",
                (br_id, sensor_name)
            )

            connection_result = cursor.fetchone()

            if connection_result:

                for cname_en, cname_en in STATUS_FA.items():
                    if cname_en == connection_result:
                        cname_fa = cname_fa[0]
                        break
                else:
                    pass  # Todo error handeling

                template = answer_templates[intent]["final"].format(barracks_name_1=barracks_name,
                                                                    sensor_name_1= sensor_name ,sensor_typ_1=sensor_type,connection_type=cname_fa)

                return template
            else:
                template = answer_templates[intent]["connection_not_found"].format(barracks_name=barracks_name , sensor_name= sensor_name)
                return template


        except Exception as e:
            print("Error:", e)

    def sensor_frequency(self, params, intent):
        sensor_name = params["sensor_name_1"]
        sensor_type = params["sensor_type_1"]
        frequency = -1

        cursor = self.conn.cursor()

        try:
            # Fetch barracks_ID from the barracks table based on barracks_name
            cursor.execute(
                "SELECT parameters FROM sensors WHERE ID = ?",
                (sensor_name,)
            )

            sensor_spec = cursor.fetchone()
            json_dict = json.loads(sensor_spec[0])

            if "frequency" in json_dict:
                frequency = json_dict["frequency"]
            else:
                template = answer_templates[intent]["frequency_not_found"]
                return template


            if frequency>0:
                template = answer_templates[intent]["Final"].format(
                                        sensor_type_1 = sensor_type,
                                        sensor_name_1= sensor_name ,frequency=frequency)


                return template
            else:
                template = answer_templates[intent]["frequency_not_found"]
                return template


        except Exception as e:
            print("Error:", e)


    def officer_serving_barracks(self, params, intent):
        rank = self.random.choice(self.MILITARY_RANKS_FA)
        name = self.random.choice(self.first_names)
        surname = self.random.choice(self.last_names)


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
