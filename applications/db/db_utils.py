import sqlite3
import pandas as pd
import os
import json

from shared.constants.constants import *


def create_connection(DB_NAME):
    """create a database connection to the SQLite database
        specified by the DB_NAME
    :param DB_NAME: database file
    :return: Connection object or None
    """
    connection = None

    # Check if the directory exists
    if not os.path.exists(os.path.dirname(DB_NAME)):
        # If not, create it
        os.makedirs(os.path.dirname(DB_NAME))

    # Check if the SQLite database file exists
    if not os.path.exists(DB_NAME):
        # If not, create SQLite database and necessary tables
        connection = sqlite3.connect(DB_NAME)
        cursor = connection.cursor()

        # Create tables and perform initial setup if needed
        cursor.execute('CREATE TABLE IF NOT EXISTS your_table (id INTEGER PRIMARY KEY, name TEXT)')

        # Commit changes and close the connection
        connection.commit()
        connection.close()

    try:
        connection = sqlite3.connect(DB_NAME)
    except Exception as error:
        print(error)

    return connection


def create_database_tables(conn):
    print("database Opened successfully!")
    conn.execute(
        """CREATE TABLE barracks
        (ID INTEGER primary key AUTOINCREMENT,
         name varchar not null unique,
         name_fa varchar not null unique,
         longitude double not null,
         latitude double not null,
         insider boolean not null,
         type varchar not null)
         """
    )

    print("barracks table created successfully!")

    conn.execute(
        """CREATE TABLE links
        (ID1 INT not null,
         ID2 INT not null,
         type varchar not null,
         online boolean not null,
         channel varchar not null,
         primary key (ID1, ID2),
         foreign key (ID1, ID2) references barracks(ID, ID)
         )
         """
    )

    print("links table created successfully!")

    conn.execute(
        """CREATE TABLE sub_barracks
        (ID INT not null,
         sub_ID INT not null,
         
         primary key (ID, sub_ID),
         foreign key (ID, sub_ID) references barracks(ID, ID)
         )
         """
    )

    print("sub_barracks table created successfully!")

    conn.execute(
        """CREATE TABLE staffs
        (ID INTEGER primary key AUTOINCREMENT,
         barracks_ID INT not null,
         first_name varchar not null,
         last_name varchar not null,
         rank varchar not null,
         access_level varchar not null,
         active string not null,
         is_online string not null,
         foreign key (barracks_ID) references barracks(ID)
         )
         """
    )

    print("staff table created successfully!")

    conn.execute(
        """CREATE TABLE sensors
        (ID INTEGER primary key AUTOINCREMENT,
         barracks_ID INT not null,
         sensor_type varchar not null,
         parameters varchar not null,
         longitude double not null,
         latitude double not null,
         radius double not null,
         insider boolean not null,
         ip varchar not null,
         link_type varchar not null,
         online boolean not null,
         name varchar not null,
         foreign key (barracks_ID) references barracks(ID)
         )
         """
    )

    print("sensors table created successfully!")


def add_barracks_to_db(conn, name, name_fa, longitude, latitude, insider, type):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "Insert INTO barracks (name, name_fa, longitude, latitude, insider, type)\
                VALUES (?, ?, ?, ?, ?, ?)",
            (name, name_fa, longitude, latitude, insider, type),
        )
    except sqlite3.IntegrityError as e:
        print(e)
        pass
    print("Barracks added to database successfully!")
    cursor.close()


def add_link_to_db(conn, ID1, ID2, type, online, channel):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "Insert INTO links (ID1, ID2, type, online, channel)\
                VALUES (?, ?, ?, ?, ?)",
            (ID1, ID2, type, online, channel),
        )
    except sqlite3.IntegrityError as e:
        print(e)
        pass
    print("Link added to database successfully!")
    cursor.close()


def add_sub_barracks_to_db(conn, ID, sub_ID):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "Insert INTO sub_barracks (ID, sub_ID)\
                VALUES (?, ?)",
            (ID, sub_ID),
        )
    except sqlite3.IntegrityError as e:
        print(e)
        pass
    print("Sub_barracks added to database successfully!")
    cursor.close()


def add_staff_to_db(
        conn, first_name, last_name, rank, barracks_ID, access_level, active, is_online):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "Insert INTO staffs (first_name, last_name, rank, barracks_ID, access_level, active, is_online)\
                VALUES (?, ?, ?, ?, ?, ?, ?)",
            (first_name, last_name, rank, barracks_ID, access_level, active, is_online)
        )
    except sqlite3.IntegrityError as e:
        print(e)
        pass
    print("Staff added to database successfully!")
    cursor.close()

def add_sensor_to_db(
        conn,
        sensor_type,
        parameters,
        longitude,
        latitude,
        radius,
        insider,
        ip,
        barracks_ID,
        link_type,
        online,
        name
):
    cursor = conn.cursor()
    try:
        parameters_json = json.dumps(parameters)

        cursor.execute(
            "INSERT INTO sensors (sensor_type, parameters, longitude, latitude, radius, insider, ip, barracks_ID, link_type, online, name) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)",
            (
                sensor_type,
                parameters_json,
                longitude,
                latitude,
                radius,
                insider,
                ip,
                barracks_ID,
                link_type,
                online,
                name,
            )
        )
        # Set name equal to the ID of the table
        name = cursor.lastrowid
        conn.commit()  # Commit the transaction
        return name  # Return the updated name
    except Exception as e:
        print("Error:", e)
        conn.rollback()  # Rollback the transaction in case of an error

def add_sensor_to_db_dep(
        conn,
        sensor_type,
        parameters,
        longitude,
        latitude,
        radius,
        insider,
        ip,
        barracks_ID,
        link_type,
        online,
        name
):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "Insert INTO sensors (sensor_type, parameters, longitude, latitude, radius, insider, ip, barracks_ID, link_type, online, name)\
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sensor_type,
                parameters,
                longitude,
                latitude,
                radius,
                insider,
                ip,
                barracks_ID,
                link_type,
                online,
                name
            )
        )
    except sqlite3.IntegrityError as e:
        print(e)
        pass
    print("Sensor added to database successfully!")
    cursor.close()


def get_all_baracks() -> pd.DataFrame:
    conn = create_connection(DB_NAME)
    sql_command: str = """SELECT * FROM barracks"""
    df: pd.DataFrame = pd.read_sql_query(sql_command, conn)  # type: ignore
    conn.commit()  # type:ignore
    conn.close()  # type:ignore
    return df


def get_all_staffs():
    conn = create_connection(DB_NAME)
    sql_command: str = """SELECT * FROM staffs"""
    df: pd.DataFrame = pd.read_sql_query(sql_command, conn)  # type: ignore
    conn.commit()  # type:ignore
    conn.close()  # type:ignore
    return df


def get_all_sensors():
    conn = create_connection(DB_NAME)
    sql_command: str = """SELECT * FROM sensors"""
    df: pd.DataFrame = pd.read_sql_query(sql_command, conn)  # type: ignore
    conn.commit()  # type:ignore
    conn.close()  # type:ignore
    return df


def get_baracks_by_kwargs(**kwargs):
    conn = create_connection(DB_NAME)
    constraint: str = " AND ".join([k + "=" + "'" + v + "'" for k, v in kwargs.items()])
    sql_command: str = f"""SELECT * FROM barracks where {constraint}"""
    df: pd.DataFrame = pd.read_sql_query(sql_command, conn)  # type: ignore
    conn.commit()  # type:ignore
    conn.close()  # type:ignore
    return df


def get_staff_by_kwargs(**kwargs):
    conn = create_connection(DB_NAME)
    constraint: str = " AND ".join([k + "=" + "'" + v + "'" for k, v in kwargs.items()])
    sql_command: str = f"""SELECT * FROM staff where {constraint}"""
    df: pd.DataFrame = pd.read_sql_query(sql_command, conn)  # type: ignore
    conn.commit()  # type:ignore
    conn.close()  # type:ignore
    return df


def get_sensors_by_kwargs(**kwargs):
    conn = create_connection(DB_NAME)
    constraint: str = " AND ".join([k + "=" + "'" + v + "'" for k, v in kwargs.items()])
    sql_command: str = f"""SELECT * FROM sensors where {constraint}"""
    df: pd.DataFrame = pd.read_sql_query(sql_command, conn)  # type: ignore
    conn.commit()  # type:ignore
    conn.close()  # type:ignore
    return df

# get_baracks_by_kwargs(name="Qom")
# if __name__ == "__main__":

#     get_all_baracks()

#     conn = sqlite3.connect(DB_NAME)
#     create_database_tables(conn)


#     # add_barracks_to_db(conn, "Command Center", 35.710565, 51.422119, True, BARRACKS_TYPE[0])
#     # add_barracks_to_db(conn, "Qom", 34.561169367937204, 50.82368163929708, True, BARRACKS_TYPE[2])
#     # add_link_to_db(conn, 1, 2, CONNECTION_TYPE[0], True, CHNNELS_TYPE[0])
#     # add_sub_barracks_to_db(conn, 1, 2)  # Barracks 1 is a sub-barracks of Barracks 2
#     # add_sensor_to_db(conn, SENSORS_TYPE[1], '{}', 34.503267003464394, 50.863662298489274, 10.0, True, "173.156.11.1", 2, CONNECTION_TYPE[0], True)
#     # add_staff_to_db(conn, "سعید", "بی‌باک", "General", 1, "Admin", True)
#     conn.commit()
#     conn.close()
