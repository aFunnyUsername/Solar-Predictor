import requests
import os
from datetime import datetime, timezone, timedelta

url = 'https://graphical.weather.gov/xml/sample_products/browser_interface/ndfdXMLclient.php'
def retrieve_data(latitude, longitude, units, data, file_path_new, file_path_old):
	now = datetime.now(timezone(-timedelta(hours=5))) 
	day_later = now + timedelta(days=1)
	payload = {'lat': latitude, 'lon': longitude, 'begin': now.isoformat(), 'end': day_later.isoformat(), 'Unit': units, 'One or more ndfd elements': data}  
	ret = requests.get(url, params=payload)
	open(file_path_new, 'wb').write(ret.content)
	if os.path.getmtime(file_path_new) > os.path.getmtime(file_path_old):
		open(file_path_old, 'wb').write(ret.content)	
		return True
	else:
		return False                                                                           
#retrieve_data('44.883', '-93.233', 'e', 'sky=sky', "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\data_nws.xml", "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\data_nws_old.xml")












