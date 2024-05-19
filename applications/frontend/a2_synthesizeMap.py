# import the library
# import sys
# import webbrowser

import folium

from applications.db import db_utils
from shared.constants.constants import *

class Map:
    def __init__(self, center, zoom_start, barracks_df, sensors_df):
        self.center = center
        self.zoom_start = zoom_start
        self.barracks = barracks_df
        self.sensors = sensors_df
        # add marker one by one on the map
        self.map_ = folium.Map(location=self.center, zoom_start=self.zoom_start)

    def save_map(self, path_target):
        self.map_.save(path_target)
    def make_map(self):
        # Create the map
        my_map = folium.Map(location=self.center, zoom_start=self.zoom_start)


        # add marker one by one on the map
        for i in range(0, len(self.barracks)):
            html = f"""
                <h1> {self.barracks.iloc[i]['name']}</h1>
                <p>Details:
                <ul>
                    <li>insider: {'Yes' if self.barracks.iloc[i]['insider']==1 else 'No'}</li>
                    <li>type: {self.barracks.iloc[i]['type']}</li>
                </ul>
                <a> <i class="fa fa-fort-awesome" aria-hidden="true"></i> </a>
                </p>
                """
            iframe = folium.IFrame(html=html, width=200, height=200)
            popup = folium.Popup(iframe, max_width=265)

            # Replace (x, y) with the width and height of your icon image
            barracks_icon = folium.CustomIcon(barracks_image_icon, icon_size=(48, 48))

            marker = folium.Marker(location=[
                    self.barracks.iloc[i]["latitude"],
                    self.barracks.iloc[i]["longitude"],
                ], icon=barracks_icon, tooltip=f"<h4> {self.barracks.iloc[i]['name_fa']}</h4>",popup=popup, icon_size=(32, 32))

            marker.add_to(my_map)

        for index, sensor in self.sensors.iterrows():
            # print(type(sensor['barracks_ID']))
            # df.loc[condition].head(1)
            html = f"""
                <h1> {self.barracks.loc[self.barracks['ID'] == sensor.barracks_ID].iloc[0]['name']}: {sensor.ip}</h1>
                <p>Details:
                <ul>
                    <li>radius: {sensor['sensor_type']}</li>
                    <li>radius: {sensor['radius']}</li>
                    <li>insider: {'Yes' if sensor['insider'] == 0 else 'No'}</li>
                    <li>ip: {sensor['ip']}</li>
                    <li>link type: {sensor['link_type']}</li>
                    <li>online: {'Yes' if sensor['online'] == 1 else 'No'}</li>
                    <li>params: {""}</li>
                </ul>
                </p>
                """
            # <a> <i class="fa fa-fort-awesome" aria-hidden="true"></i> </a>
            iframe = folium.IFrame(html=html, width=300, height=300)
            popup = folium.Popup(iframe, max_width=300)

            radar_icon = folium.CustomIcon(radar_image_icon, icon_size=(32, 32))

            icon_image = 'antenna.png'
            antenna_icon = folium.CustomIcon(antenna_image_icon, icon_size=(32, 32))

            icon_image = 'jammer.png'
            jammmer_icon = folium.CustomIcon(jammer_image_icon, icon_size=(32, 32))

            sensor_icon = jammmer_icon
            if sensor['sensor_type'] == 'Radar':
                sensor_icon = radar_icon
            elif sensor['sensor_type'] == 'Antenna':
                sensor_icon = antenna_icon

            marker = folium.Marker(location=[
                sensor["latitude"],
                sensor["longitude"],
            ], icon=sensor_icon, tooltip=f"<h4>{sensor['sensor_type']}- {sensor['ip']}</h4>", popup=popup,
                icon_size=(32, 32))
            marker.add_to(my_map)
            # print(
            #     sensor,
            #     self.barracks.loc[self.barracks['ID'] == sensor.barracks_ID].head(1),
            #     self.barracks.loc[self.barracks['ID'] == sensor.barracks_ID].iloc[0]['latitude'],
            #     self.barracks.loc[self.barracks['ID'] == sensor.barracks_ID].iloc[0]['longitude'],
            #     sensor["latitude"],
            #     sensor["longitude"])
        self.map_ = my_map


barracks_df = db_utils.get_all_baracks()
coords = [barracks_df["latitude"].mean(), barracks_df["longitude"].mean()]
sensors_df = db_utils.get_all_sensors()

map = Map(center=coords, zoom_start=6, barracks_df=barracks_df, sensors_df=sensors_df)
map.make_map()
# Display the map
map.save_map(synthetic_map_html)
