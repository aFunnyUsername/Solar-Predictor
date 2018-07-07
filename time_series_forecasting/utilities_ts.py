#general ML utilities like reading data, splitting into X and Y, splitting into train/test
#normalizing, standardizing, getting statistics, etc.

import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing as pp
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt


#this function will create a datetime column out of the date and time columns
#currently present in the tmy dataset.  This is to make it consistent with
#pandas parsing of datetimes
def parse_index_time_column(fp, variables):
	tmy = pd.read_csv(fp, header=1, low_memory=False)
	time_list = []
	for i, time in enumerate(tmy['Time (HH:MM)']):
		if time == '24:00:00':
			time = '0:00'
			time_list.append(time)
		else:
			time_list.append(time)
	#This is done because tmy stores its time data from hour 1 to hour 24 but we
	#need it from hour 0 to hour 23.
	time_list.insert(0, time_list.pop(len(time_list) - 1))
	tmy['Time (HH:MM)'] = time_list

	datetime_list = []
	for i, date in enumerate(tmy['Date (MM/DD/YYYY)']):
		datetime = date + ' ' + time_list[i]
		datetime_list.append(datetime)
	
	tmy['DateTime'] = datetime_list
	tmy['DateTime'] = pd.to_datetime(tmy['DateTime'])
	
	tmy_reduced = pd.DataFrame()
	tmy_reduced['Month'] = [tmy.loc[i, 'DateTime'].month for i in range(len(tmy['DateTime']))]
	tmy_reduced['Day'] = [tmy.loc[i, 'DateTime'].day for i in range(len(tmy['DateTime']))]
	tmy_reduced['Hour'] = [tmy.loc[i, 'DateTime'].hour for i in range(len(tmy['DateTime']))]
	for variable in variables:
		tmy_reduced[variable] = tmy[variable].values
	
	return tmy_reduced

def split_dataset_validation(df, validation_entries):
	split_point = df.shape[0] - validation_entries
	dataset_df, validation_df = df.loc[:split_point], df.loc[split_point:]	
	return dataset_df, validation_df

def split_dataset_train_test(df, values, split):
	X = df[values].values
	X = X.astype('float32')
	train_size = int(X.shape[0] * split)
	train, test = X[0:train_size], X[train_size:]
	return train, test

def naive_baseline(train_dataset, test_dataset):
	#NOTE, this might just be a walk-forward validation function later, if it's
	#similar enough when we actually train/make predictions
	history = [x for x in train_dataset]
	predictions = list()	
	for i in range(len(test_dataset)):
		#prediction
		yhat = history[-1]
		#NOTE, for the first value, this will be the last value of the training
		#dataset.  As we add observations, it will become the observation at
		#t-1.  Pretty neat.
		predictions.append(yhat)
		#observation
		#NOTE, what actually happend
		obs = test_dataset[i] 
		history.append(obs)
		#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	rmse = sqrt(mean_squared_error(test_dataset, predictions))
	return rmse

def groupby_things(df, frequency, column):
	groups = df.groupby(frequency)[column]
	times = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
	times = pd.DataFrame(times)
	times.columns = range(0, times.shape[1])
	return times

def timely_line_plots(df, frequency, column):
	plot_df = groupby_things(df, frequency, column)
	column = column[:4]
	plot_df.plot(subplots=True, legend=False)
	plt.savefig("plots\\" + column + "_" + frequency + "_line.png")
	plt.close()

"""def timely_histogram_plots(df, frequency, column):
	plot_df = groupby_things(df, frequency, column)
	column = column[:4]
	plot_df.hist(subplots=True, legend=False)
	plt.savefig("plots\\" + column + "_" + frequency + "_hist.png")
	plt.close()"""

def timely_density_plots(df, frequency, column):
	plot_df = groupby_things(df, frequency, column)
	column = column[:4]
	plot_df.plot(subplots=True, legend=False, kind='kde')
	plt.savefig("plots\\" + column + "_" + frequency + "_dense.png")
	plt.close()

def timely_boxwhisker_plots(df, frequency, column):
	plot_df = groupby_things(df, frequency, column)
	column = column[:4]
	plot_df.boxplot()
	plt.savefig("plots\\" + column + "_" + frequency + "_box.png")
	plt.close()

def seasonality(df, frequency, steps, column):
	X = df[column].values
	diff = list()
	for i in range(steps, len(X)):
		value = X[i] = X[i - steps]
		diff.append(value)
	diff_series = pd.Series(diff)	
	difference_df = df.loc[steps:]
	difference_df.replace(difference_df[column], diff_series, inplace=True)
	plot_df = groupby_things(difference_df, frequency, column)
	plot_df.plot(subplots=True, legend=False)	
	column = column[:4]
	plt.savefig("plots\\" + column + "_" + frequency + "_season.png")





