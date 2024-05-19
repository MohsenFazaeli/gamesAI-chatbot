import pandas as pd
import os.path as opath

from shared.constants.a1_general import *

# DB configs
DB_NAME = f".storage/db/C4I_system-{stage}.db"



resources_path = opath.join('shared', 'resources')

cities = pd.read_csv(opath.join(resources_path,"irancities.csv"))
first_names = pd.read_excel(opath.join(resources_path,"farsi_names.xlsx"))
last_names = pd.read_csv(opath.join(resources_path,"last_names.csv"))


# Params EN FA
BARRACKS_TYPE = ["ADOC", "SOC", "ROC"]
BARRACKS_TYPE_FA = {"ADOC", "SOC", "ROC"}

CONNECTION_TYPE = ["Fiber", "Radio", "Satellite"]
CONNECTION_TYPE_FA = {
                      "Fiber": ["فیبر"],
                      "Radio": ["رادیو", "رادیوئی"],
                      "Satellite": ["ماهواره", "ماهواره‌ای", "فضایی"]}

CHNNELS_TYPE = [
    "Unknown",
    "Arman",
    "Soroush",
    "Farzin",
]  # Only for Radio
CHNNELS_TYPE_FA = {
    "Unknown": ["نامشخص"],
    "Arman": ["آرمان"],
    "Soroush": ["سروش"],
    "Farzin": ["فرزین"],
}  # Only for Radio

SENSORS_TYPE = ["Radar", "Antenna", "Jammer"]
SENSORS_TYPE_FA = {"None": ["سنسور"],
                   "Sensor": ["سنسور"],
                   "Radar": ["رادار"],
                   "Antenna": ["آنتن"],
                   "Jammer": ["جمر"]}

MILITARY_RANKS = [
    "General",
    "Colonel",
    "Major",
    "Captain",
    "Lieutenant",
    "Sergeant",
    "Corporal",
    "Private",
]
MILITARY_RANKS_FA = {
    "General": ["ارتشبد"],
    "Colonel": ["سرتیپ"],
    "Major": ["سرهنگ"],
    "Captain": ["سروان"],
    "Lieutenant": ["ستوان"],
    "Sergeant": ["گروهبان"],
    "Corporal": ["سرجوخه"],
    "Private": ["سرباز"],
}

ACCESS_LEVEL = ["Admin", "User"]
ACCESS_LEVEL_FA = {"Admin": "مدیر", "User": "کاربر"}

STATUS = ["ACTIVE", "INACTIVE", "ONLINE", "OFFLINE"]
STATUS_FA = {"ACTIVE": ["فعال"],
             "INACTIVE": ["غیر فعال"],
             "ONLINE": ["آنلاین"],
             "OFFLINE": ["آفلاین"]}

DIRECTIONS = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest", "Central"]
DIRECTIONS_FA = {
    "North": ["شمال"],
    "Northeast": ["شمال شرق"],
    "East": ["شرق"],
    "Southeast": ["جنوب شرق"],
    "South": ["جنوب"],
    "Southwest": ["جنوب غرب"],
    "West": ["غرب"],
    "Northwest": ["شمال غرب"],
    "Central": ["مرکزی"]
}
