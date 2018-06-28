import os
from xml.etree import ElementTree as ET
import numpy
import pandas
import time

total_clouds_df = pandas.DataFrame()

def read_xml(file_path):
	future_clouds_df = pandas.DataFrame()
	full_file = file_path 
	dom = ET.parse(full_file)
	root = dom.getroot()
	#print(dom)
	#print(root.tag, root.attrib)
	for thing in root:
		if thing.tag == 'data':
			data = thing

	for table in data:
		#print(table.tag, table.attrib)
		if table.tag == 'parameters':
			parameters = table

	for weather in parameters:
		if weather.tag == 'temperature':
			if weather.attrib['type'] == 'hourly':
				temp = weather
		if weather.tag == 'cloud-amount':
			if weather.attrib['type'] == 'total':
				clouds = weather
		if weather.tag == 'humidity':
			#print(weather.attrib)
			if weather.attrib['type'] == 'relative':	
				#print(weather.attrib['type'])
				humid = weather
 
	cloud_list = []
	humid_list = []
	temp_list = []
	for element in clouds:
		if element.text != 'Cloud Cover Amount':
			element = int(element.text) / 10
			cloud_list.append(element)
		
	for element in humid:
		if element.text != 'Relative Humidity':
			humid_list.append(element.text)

	for element in temp:
		if element.text != 'Temperature':
			celsius = (int(element.text) - 32) * (5 / 9)		
			temp_list.append(int(celsius))
	#print(clouds.tag, clouds.attrib)
	#print(cloud_list)
	#print(len(cloud_list))
	#print(len(humid_list))
	#print(clouds.attrib.get('time-layout'))
	time_key = clouds.attrib.get('time-layout')
	poss_list = []
	key_loc_list = []

	time_list_raw = []
	time_list = []
	date_list = []

	def parse_timestamp(dt):
		year = dt[:4]
		month = dt[5:7]
		day = dt[8:10]
		hour = dt[11:13]
		date = month + '/' + day + '/' + year	
		time = hour + ':00'
		return [date, time]

	for table in data:
		if table.tag == 'time-layout':
			#print(table.tag, table.attrib)
			time_layouts = table
			for poss in time_layouts:
					poss_list.append(poss)
	for poss in poss_list:
			if poss.tag == 'layout-key':
				key_loc_list.append(poss_list.index(poss))
				check_key = poss.text
				if check_key == time_key:
					correct_key_loc = poss_list.index(poss)

	next_key_loc = key_loc_list.index(correct_key_loc) + 1
	next_key = key_loc_list[next_key_loc]
	time_list_elements = poss_list[correct_key_loc + 1:next_key]

	for time in time_list_elements:
		time_list_raw.append(time.text)

	for dt in time_list_raw:
		split_dt = parse_timestamp(dt)
		date_list.append(split_dt[0])
		time_list.append(split_dt[1])		
		
	#print(date_list)
	#print(time_list)

	future_weather_df = pandas.DataFrame({'Date': date_list, 'Time' : time_list, 'Cloud Coverage' : cloud_list, 'Temperature' : temp_list, 'Relative Humidity' : humid_list})
	return future_weather_df

#keep just in case this doesn't work
"""def write_to_csv(new_df):
	filename_new = 'data_csv_nws.csv'
	filename_old = 'data_csv_nws_old.csv'
	old_df = pandas.read_csv(filename_old)
	#print(old_df, new_df)
	merged_df = pandas.merge(old_df, new_df, on = ['Date', 'Time'], how='outer')
	#print(merged_df)
	index_frame_merged_df = pandas.notnull(merged_df)
	#print(index_frame_merged_df)

	clouds_x = merged_df['Cloud Coverage_x'].values
	clouds_y = merged_df['Cloud Coverage_y'].values 
	humid_x = merged_df['Relative Humidity_x'].values
	humid_y = merged_df['Relative Humidity_y'].values
	temp_x = merged_df['Temperature_x'].values
	temp_y = merged_df['Temperature_y'].values	
	i = 0

	for value in index_frame_merged_df['Cloud Coverage_y']:
		if value:
			clouds_x[i] = clouds_y[i]
		i += 1

	i = 0
	for value in index_frame_merged_df['Relative Humidity_y']:
		if value:
			humid_x[i] = humid_y[i]
		i += 1

	i=0
	for value in index_frame_merged_df['Temperature_y']:
		if value:
			temp_x[i] = temp_y[i]
		i += 1
	
	new_times = merged_df['Time'].values
	new_dates = merged_df['Date'].values

	write_df = pandas.DataFrame({'Date': new_dates, 'Time': new_times, 'Cloud Coverage': clouds_x, 'Temperature': temp_x, 'Relative Humidity': humid_x})
	print(write_df)	
	write_df.to_csv(filename_new)
#['Date', 'Time', 'Cloud Coverage', 'Temperature', 'Relative Humidity']
	if os.path.getmtime(filename_new) > os.path.getmtime(filename_old):
		write_df.to_csv(filename_old)
		return True
	else:
		return False
"""






