"""
Generates random samples for database tables in a reasonable range
"""
import os
print(os.getcwd())

import sys
print(sys.path)

import random
from geopy import distance

from applications.db.db_utils import *
from shared.constants.constants import *



# Function to generate random parameters for antennas
def generate_antenna_parameters():
    parameters = {
        'frequency': random.randint(100, 1000),
        'gain': round(random.uniform(10, 50), 2),
        'polarization': random.choice(['vertical', 'horizontal']),
        'bandwidth': round(random.uniform(1, 10), 2)
    }
    return parameters

# Function to generate random parameters for jammers
def generate_jammer_parameters():
    parameters = {
        'frequency': random.randint(500, 2000),
        'power': round(random.uniform(1, 50), 2),
        'jamming_technique': random.choice(['noise', 'spoofing', 'deception']),
        'effectiveness': round(random.uniform(0, 100), 2)
    }
    return parameters

# Function to generate random parameters for radars
def generate_radar_parameters():
    parameters = {
        'frequency': random.randint(1000, 5000),
        'range_resolution': round(random.uniform(0.1, 10), 2),
        'maximum_range': random.randint(100, 1000),
        'antenna_height': random.randint(5, 50)
    }
    return parameters

def generate_sensor_params(type):
    #["Radar", "Antenna", "Jammer"]
    if type == 'Antenna':
        return generate_antenna_parameters()
    if type == 'Radar':
        return generate_radar_parameters()

    return generate_jammer_parameters()


# Generate seed_data_size sample barracks rows
#cities = pd.read_csv(r"app\\resources\\irancities.csv")


types = ["SOC", "ROC"]  # excluded "ADOC" to make sure only one "ADOC" is allowed
insiders = [True, False]
cities_list = cities.iloc[random.sample(range(0, len(cities)), seed_data_size)]
cities_list.reset_index(drop=True, inplace=True)
types_list = [random.randint(0, len(types) - 1) for _ in range(seed_data_size)]
insider_list = [random.randint(0, 1) for _ in range(seed_data_size)]

if os.path.exists(DB_NAME):
    print(DB_NAME, "removed")
    os.remove(DB_NAME)

conn = create_connection(DB_NAME)
create_database_tables(conn)


for index, row in cities_list.iterrows():
    name = row["city"]
    name_fa = row["city_FA"]
    longitude = row["lng"]
    latitude = row["lat"]
    type = types[types_list[index]]  # type: ignore
    insider = insiders[insider_list[index]]  # type: ignore
    add_barracks_to_db(conn, name, name_fa, longitude, latitude, insider, type)
conn.commit()
conn.close()

# Generate seed_data_size sample links rows
conn = create_connection(DB_NAME)
barracks = get_all_baracks()[["ID", "longitude", "latitude"]]
onlines = [False, True]
types = ["Fiber", "Radio", "Satellite"]
channels = ["Unknown"]
link_ids = [random.sample(barracks["ID"].to_list(), 2) for _ in range(seed_data_size)]
online_list = [random.randint(0, 1) for _ in range(seed_data_size)]
channel_list = [random.randint(0, len(channels) - 1) for _ in range(seed_data_size)]
for index in range(len(link_ids)):
    id1 = link_ids[index][0]
    id2 = link_ids[index][1]
    coords_1 = (
        barracks[barracks["ID"] == id1]["latitude"].values[0],
        barracks[barracks["ID"] == id1]["longitude"].values[0],
    )
    coords_2 = (
        barracks[barracks["ID"] == id2]["latitude"].values[0],
        barracks[barracks["ID"] == id2]["longitude"].values[0],
    )
    dist = distance.geodesic(coords_1, coords_2).km
    type = (
        "Radio"
        if dist <= 50
        else "Fiber"
        if dist >= 50 and dist <= 100
        else "Satellite"
    )  # check validity
    online = onlines[online_list[index]]
    channel = channels[channel_list[index]]
    add_link_to_db(conn, id1, id2, type, online, channel)
conn.commit()
conn.close()


for _ in range(1):
    # Generate seed_data_size sample sensors
    conn = create_connection(DB_NAME)
    barracks = get_all_baracks()
    onlines = [False, True]
    types = ["Radar", "Antenna", "Jammer"]
    link_types = ["Fiber", "Radio", "Satellite"]
    insiders = [True, False]
    max_sensor_radius = 200
    min_sensor_radius = 5
    radiuses = [
        random.randint(min_sensor_radius, max_sensor_radius) for _ in range(seed_data_size)
    ]
    channels = ["Unknown"]
    parameter = "{}"
    barracks_ids = [
        random.sample(barracks["ID"].to_list(), 1) for _ in range(seed_data_size)
    ]
    type_list = [random.randint(0, len(types) - 1) for _ in range(seed_data_size)]
    link_type_list = [random.randint(0, len(link_types) - 1) for _ in range(seed_data_size)]
    insider_list = [random.randint(0, len(insiders) - 1) for _ in range(seed_data_size)]
    online_list = [random.randint(0, 1) for _ in range(seed_data_size)]
    ips = [
        ".".join(map(str, (random.randint(0, 255) for _ in range(4))))
        for _ in range(seed_data_size)
    ]
    for index in range(len(barracks_ids)):
        sensor_type = types[type_list[index]]
        link_type = link_types[link_type_list[index]]
        online = onlines[online_list[index]]
        parameters = generate_sensor_params(sensor_type)
        latitude = barracks.loc[
            barracks["ID"] == barracks_ids[index][0], "latitude"
        ].values[0]
        longitude = barracks.loc[
            barracks["ID"] == barracks_ids[index][0], "longitude"
        ].values[0]
        insider = insiders[insider_list[index]]  # type: ignore
        #barracks_ID = barracks[barracks["ID"] == barracks_ids[index][0]]["ID"].values[0]
        barracks_ID = int(barracks[barracks["ID"] == barracks_ids[index][0]]["ID"].values[0])

        ip = ips[index]
        radius = float(radiuses[index])
        add_sensor_to_db(
            conn,
            "Radar",
            parameters,
            longitude,
            latitude,
            radius,
            insider,
            ip,
            barracks_ID,
            link_type,
            online,
            ""
        )
        add_sensor_to_db(
            conn,
            "Antenna",
            parameters,
            longitude,
            latitude,
            radius,
            insider,
            ip,
            barracks_ID,
            link_type,
            online,
            ""
        )
        add_sensor_to_db(
            conn,
            "Jammer",
            parameters,
            longitude,
            latitude,
            radius,
            insider,
            ip,
            barracks_ID,
            link_type,
            online,
            ""
        )
    conn.commit()
    conn.close()



# Generating seed_data_size sub-barrakcs
conn = create_connection(DB_NAME)
barracks = get_all_baracks()
barracks_ids = get_all_baracks()["ID"].to_list()
barracks_ids = [random.sample(barracks_ids, 2) for _ in range(seed_data_size)]
for index in range(len(barracks_ids)):
    higher_barracks = barracks[barracks["ID"] == barracks_ids[index][0]]
    sub_barracks = barracks[barracks["ID"] == barracks_ids[index][1]]
    barracks_type = higher_barracks["type"].values[0]
    sub_barracks_type = sub_barracks["type"].values[0]
    if sub_barracks_type == "ADOC":
        continue
    if sub_barracks_type == "ROC" and barracks_type not in ["ADOC", "SOC"]:
        continue
    add_sub_barracks_to_db(
        conn, int(higher_barracks["ID"].values[0]), int(sub_barracks["ID"].values[0])
    )
conn.commit()
conn.close()


# Generating seed_data_size Staff members
conn = create_connection(DB_NAME)
barracks_ids = get_all_baracks()["ID"].to_list()
barracks_ids = [random.sample(barracks_ids, 1) for _ in range(seed_data_size)]

ranks = [
    "General",
    "Colonel",
    "Major",
    "Captain",
    "Lieutenant",
    "Sergeant",
    "Corporal",
    "Private",
]
ranks_list = [random.randint(0, len(ranks) - 1) for _ in range(seed_data_size)]

access_levels = ["Admin", "User"]
access_levels_list = [
    random.randint(0, len(access_levels) - 1) for _ in range(seed_data_size)
]

actives = [True, False]
active_list = [random.randint(0, len(actives) - 1) for _ in range(seed_data_size)]
online_list = [random.randint(0, len(actives) - 1) for _ in range(seed_data_size)]


first_name_list = [random.randint(0, len(first_names)) for _ in range(seed_data_size)]
last_names_list = [random.randint(0, len(last_names)) for _ in range(seed_data_size)]


for index in range(len(barracks_ids)):
    for _ in range(random.randint(0, 20)):
        barracks_ID = barracks_ids[index][0]
        rank = ranks[ranks_list[index]]
        is_active = actives[active_list[index]]
        is_online = actives[online_list[index]]

        access_level = access_levels[access_levels_list[index]]
        first_name = first_names.at[random.randint(0, len(first_names)-1), "farsi_names"]
        last_name = last_names.at[ random.randint(0, len(last_names)-1) , "last_name"]
        print( first_name, last_name, rank, barracks_ID, access_level, is_active, is_online)
        add_staff_to_db(
            conn, first_name, last_name, rank, barracks_ID, access_level, is_active, is_online
        )
conn.commit()
conn.close()
