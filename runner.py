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
DNI_model_fp = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\models\\SVR_DNI_final_06272018.sav"
DHI_model_fp = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\models\\SVR_DHI_final_06272018.sav"
write_df_fp = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\prediction_csvs\\SVR_predictions.csv"
#NOTE, for now, just make sure than write_df_fp matches the model that we're using but we'll do it smarter some other time


def main():
	now = get_now()[0]
	if ret.retrieve_data(lat, lon, unit, points, file_path_new, file_path_old):
		print('GREAT! XML READER EXECUTING...')

		read_output_df = read.read_xml(file_path_new)			
		future_df = pred.predictor(read_output_df, DNI_model_fp, DHI_model_fp)
	
		print(future_df)

		print(pred.write_df(write_df_fp, future_df))	
		print('csv updated at: ' + str(now))	
	else:
		#NOTE, later I will have the ndfd reader actually return a false in the correct place
		print('error in reading from ndfd')


main()
scheduler = BlockingScheduler(standalone=True)
scheduler.add_job(main, 'interval', hours=1, misfire_grace_time = 60)
try:
	scheduler.start()
except (KeyboardInterrupt, SystemExit):
	print('fsaklj fuck exiting...')








