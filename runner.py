import ndfdAPI as ret
import readXML as read
import predictor as pred
import os
from apscheduler.schedulers.blocking import BlockingScheduler
import time
from datetime import datetime, timezone, timedelta
import sys

def get_now():
	thing = datetime.now(timezone(-timedelta(hours=5)))
	remove_space = str(thing).replace(' ', '_')
	remove_colons = remove_space.replace(':', 't')
	remove_period = remove_colons.replace('.', 'p')	
	return (thing, remove_period)

lat = '44.883'
lon = '-93.233'
unit = 'e'
points = 'sky=sky'
to_predict = ['DNI', 'DHI']
file_path_new = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\data_nws.xml"
file_path_old = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\data_nws_old.xml"
tmy_file_path = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\tmy3MSPairport.csv" 
csv_file_path_new = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\data_csv_nws.csv"
csv_file_path_old = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\data_csv_nws_old.csv"
DNI_model_fp = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\SVR_DNI_final_06252018.sav"
DHI_model_fp = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\SVR_DHI_final_06252018.sav"
write_df_fp = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\prediction_csvs\\pred_" + get_now()[1] + "_.csv"


def main():
	now = get_now()[0]
	if ret.retrieve_data(lat, lon, unit, points, file_path_new, file_path_old):
		print('GREAT! XML READER EXECUTING...')

		read_output = read.read_xml(file_path_new)
		if read.write_to_csv(read_output):
			print('Great!  CSV updated at: ' + str(now))
			
			future_data = pred.predictor(csv_file_path_new, DNI_model_fp, DHI_model_fp)
			print(future_data)
			predictions_list = []

			i = 0
			for thing in future_data:
				if i == 0:
					i += 1
					continue
				predictions_list.append(thing)
				i += 1

			future_df = pred.update_df(predictions_list, future_data[0], to_predict) 			
			print(future_df)

			pred.write_df("C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\prediction_csvs\\pred_" + get_now()[1] + "_.csv", future_df)
		else:
			print('fuck')

	else:
		print('fuck')

main()
scheduler = BlockingScheduler(standalone=True)
scheduler.add_job(main, 'interval', hours=1)
try:
	scheduler.start()
except (KeyboardInterrupt, SystemExit):
	print('fsaklj fuck exiting...')








