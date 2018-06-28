#1 - Import Libraries
import sys
#---------------
import scipy
#---------------
import numpy as np
#---------------
import pandas as pd
from pandas.tools.plotting import scatter_matrix
#---------------
from sklearn import model_selection
from sklearn import preprocessing as pp
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.externals import joblib
#-------------------------------------------------------------------

filename_tmy = "C:\\Users\\Jake\\Desktop\\career\\Coding\\solar_prediction\\tmy3MSPairport.csv"

tmy = pd.read_csv(filename_tmy, header=1, low_memory=False)
tmy_weather = tmy[['ETR (W/m^2)', 'GHI (W/m^2)', 'DNI (W/m^2)', 'DHI (W/m^2)', 
									 'TotCld (tenths)', 'Dry-bulb (C)', 'RHum (%)']]
#create np arrays for each attribute.  Will make data wrangling later easier
ETR_a = np.array(tmy_weather['ETR (W/m^2)'].values, dtype=np.float)
GHI_a = np.array(tmy_weather['GHI (W/m^2)'].values, dtype=np.float)
DNI_a = np.array(tmy_weather['DNI (W/m^2)'].values, dtype=np.float)
DHI_a = np.array(tmy_weather['DHI (W/m^2)'].values, dtype=np.float)
TotClds_a = np.array(tmy_weather['TotCld (tenths)'].values, dtype=np.float)
DryBulb_a = np.array(tmy_weather['Dry-bulb (C)'].values, dtype=np.float)
RHum_a = np.array(tmy_weather['RHum (%)'].values, dtype=np.float)

tmy_weather_a = tmy_weather.values
#X values are all rows in ETR (sort of like time of day), Clouds, Temp, Relative Humidity
X = tmy_weather_a[:, [0, 4, 5, 6]]
#Y values are just DNI and DHI: direct and diffuse irradiance
Y = tmy_weather_a[:, [2, 3]]
#split once further:
DNI_Y = Y[:, 0]
DHI_Y = Y[:, 1]

#now, we'll split into our train and test sets.  NOTE, the reason for this is that we will be doing 
#the initial scaling on the test set, and the min and maxes found for this function will need to be
#used for scaling throughout the project (validation and prediction).  In addition, we obviously 
#won't have any Y values to create a new normalizer with predictive data, so we need to make a 
#normalizer for that globally
validation_size = 0.20
#used for RNG things that are important for machine learning 
DNI_seed = 7
DHI_seed = 5
#train_test_split splits the data randomly into train and validation sets based on the validation size
X_train, X_val, DNI_Y_train, DNI_Y_val = model_selection.train_test_split(X, DNI_Y, test_size=validation_size, random_state=DNI_seed)
X_train, X_val, DHI_Y_train, DHI_Y_val = model_selection.train_test_split(X, DHI_Y, test_size=validation_size, random_state=DHI_seed)
#NOTE that we do a separate split for each Y value we are trying to predict.  The X data is the same
#which is partially why we do 2 splits, so that the values are randomized for the second prediction

#now we will scale the training data.  scaling for further steps (validation and prediction) will
#be done in those functions
X_normalizer = pp.MinMaxScaler()
DNI_Y_normalizer = pp.MinMaxScaler()
DHI_Y_normalizer = pp.MinMaxScaler()
#NOTE each "set" gets its own normalizer since they each have their own min/maxes
#NOTE we also need to reshape the Y arrays since there are issues with 1d arrays being used with
#minmaxscaler I guess
DNI_Y_train = np.array(DNI_Y_train).reshape((len(DNI_Y_train), 1))
DHI_Y_train = np.array(DHI_Y_train).reshape((len(DHI_Y_train), 1))
	
X_train_scaled = X_normalizer.fit_transform(X_train)
DNI_Y_train_scaled = DNI_Y_normalizer.fit_transform(DNI_Y_train)
DHI_Y_train_scaled = DHI_Y_normalizer.fit_transform(DHI_Y_train)
#---------------------------------------------------------------------------------------------

def trainer(X_data, Y_data, seed, kfolds):
	#this function will be used to evaluate different models with this data
	#NOTE the y values we are evaluating will depend on which set we pass in (DNI vs DHI)
	#for the most part we'll be interested in the MAE and R^2 but that could change in the future
	scores = ['neg_mean_absolute_error', 'r2']

	models = []
	#NOTE add new models to test here and comment out if not using
	models.append(('SVM', SVR()))
	
	#here's the storage for the stats:
	r2_mean = []
	r2_std = []
	MAE_mean = []
	MAE_std = []
	model_names = []
	
	#now loop through each model, do the k fold cross validation and store the scores in the lists
	for name, model in models:
		kfold = model_selection.KFold(n_splits=kfolds, random_state=seed)
		model_names.append(name)
		for method in scores:
			#NOTE we need to ravel the Y data since what we passed in was reshaped to scale	
			cv_results = model_selection.cross_val_score(model, X_data, Y_data.ravel(), cv=kfold, scoring=scores[scores.index(method)])
			#cv_results will be a list of kfold length.  we will take the mean and std dev of this list
			#for each scoring method in scores
			if method == 'neg_mean_absolute_error':
				MAE_mean.append(cv_results.mean())
				MAE_std.append(cv_results.std())
			elif method == 'r2':
				r2_mean.append(cv_results.mean())
				r2_std.append(cv_results.std())

	#now make the df with the results:
	results_df = pd.DataFrame({
		'R Squared Mean: ': r2_mean,
		'R Squared Std: ': r2_std,
		'MAE Mean: ': MAE_mean,
		'MAE Std: ': MAE_std,
	},
	index=model_names)
	#and return this dataframe
	return results_df
#---------------------------------------------------------------------------------------------
		
def predictor(new_df, DNI_model_fp, DHI_model_fp):
	#also read in the model's serial
	DNI_model = joblib.load(DNI_model_fp)
	DHI_model = joblib.load(DHI_model_fp)

	#the following steps are required in order to match up the ETR data for the TMY with
	#the NWS data that we've been collecting
	new_date_time_list = []
	tmy_date_time_list = []
	future_ETR = []
	i = 0
	j = 0
	k = 0
	
	def remove_year(date):
		if date[0] is '0':
			new_date = date[1:-5]
		else:
			new_date = date[:-5]
		return new_date

	def remove_date_zeros(date):
		if date[0] is '0':
			new_date = date[1:]
			if new_date[-7] is '0':
				new_date = new_date[:-7] + new_date[-6:]
		elif date[-7] is '0':
			new_date = date[:-7] + date[-6:]
		return new_date

	def remove_zero_time(time):
		if time[0] is '0':
			new_time = time[1:]
		else:
			new_time = time
		return new_time

	for time in new_df['Date']:
		combined_dt = remove_year(new_df.loc[i, 'Date']) + ' ' + remove_zero_time(new_df.loc[i, 'Time'])
		new_date_time_list.append(combined_dt)
		new_df.loc[i, 'Date'] = remove_date_zeros(new_df.loc[i, 'Date'])
		new_df.loc[i, 'Time'] = remove_zero_time(new_df.loc[i, 'Time'])
		i += 1
	for date in tmy['Date (MM/DD/YYYY)']:
		combined_dt = remove_year(tmy.loc[j, 'Date (MM/DD/YYYY)']) + ' ' + remove_zero_time(tmy.loc[j, 'Time (HH:MM)'])
		tmy_date_time_list.append(combined_dt)
		j += 1
	
	new_date_time_df = pd.DataFrame({'DateTime': new_date_time_list})
	tmy_date_time_df = pd.DataFrame({'DateTime': tmy_date_time_list, 'ETR': ETR_a})
	#the .isin() function returns True if the value of the data frame passed as a parameter exists in
	#the dataframe calling the function.  False otherwise.  Here, we're looking for the Trues and adding 
	#the corresponding ETR value to the ETR list	
	for index in tmy_date_time_df['DateTime'].isin(new_date_time_df['DateTime']):
		if index:
			future_ETR.append(tmy_date_time_df.loc[k, 'ETR'])
		k += 1
	#we will add these values to the csv data frame (and eventually a new csv)	
	new_df['ETR'] = future_ETR 
	new_X_df = new_df[['ETR', 'Cloud Coverage', 'Temperature', 'Relative Humidity']]	
	#now for the real predictions
	#NOTE, same x values, but they're in a different order in the csv
	new_X = new_X_df.values	
	new_X_scaled = X_normalizer.fit_transform(new_X)	
	#now make predictions using the model and then return the still scaled Y to be re-scaled in the global frame 
	future_DNI_Y = DNI_model.predict(new_X_scaled)
	future_DHI_Y = DHI_model.predict(new_X_scaled)	
	future_DNI_Y = np.array(future_DNI_Y).reshape((len(future_DNI_Y), 1))
	future_DHI_Y = np.array(future_DHI_Y).reshape((len(future_DHI_Y), 1))

	future_DNI_Y_inverted = DNI_Y_normalizer.inverse_transform(future_DNI_Y)
	future_DHI_Y_inverted = DHI_Y_normalizer.inverse_transform(future_DHI_Y)

	new_df['DNI'] = future_DNI_Y_inverted
	new_df['DHI'] = future_DHI_Y_inverted
	#new_df.to_csv('prediction_csvs\\test.csv')

	return new_df

def write_df(fp, new_df):
	old_df = pd.read_csv(fp)
	merged_df = pd.merge(old_df, new_df, on = ['Date', 'Time'], how='outer')
	index_frame_merged_df = pd.notnull(merged_df)	
	#here, we make an index frame to see which values are present in the new dataframe that were not
	#present in the old.  The dataframes are merged so that all values are kept, nans deleted, and 
	#column names preserved
	clouds_x = merged_df['Cloud Coverage_x'].values	
	clouds_y = merged_df['Cloud Coverage_y'].values
	humid_x = merged_df['Relative Humidity_x'].values	
	humid_y = merged_df['Relative Humidity_y'].values
	temp_x = merged_df['Temperature_x'].values	
	temp_y = merged_df['Temperature_y'].values
	etr_x = merged_df['ETR_x'].values
	etr_y = merged_df['ETR_y'].values
	dni_x = merged_df['DNI_x'].values	
	dni_y = merged_df['DNI_y'].values
	dhi_x = merged_df['DHI_x'].values	
	dhi_y = merged_df['DHI_y'].values	
	#replace old values with new, unless new is NaN and old exists
	for i, value in enumerate(index_frame_merged_df['Cloud Coverage_y']):
		if value:
			clouds_x[i] = clouds_y[i]
	for i, value in enumerate(index_frame_merged_df['Relative Humidity_y']):
		if value:
			humid_x[i] = humid_y[i]
	for i, value in enumerate(index_frame_merged_df['Temperature_y']):
		if value:
			temp_x[i] = temp_y[i]
	for i, value in enumerate(index_frame_merged_df['ETR_y']):
		if value:
			etr_x[i] = etr_y[i]
	for i, value in enumerate(index_frame_merged_df['DNI_y']):
		if value:
			dni_x[i] = dni_y[i]
	for i, value in enumerate(index_frame_merged_df['DHI_y']):
		if value:
			dhi_x[i] = dhi_y[i]

	new_times = merged_df['Time'].values
	new_dates = merged_df['Date'].values
	
	write_df = pd.DataFrame({'Date': new_dates, 
													 'Time': new_times,
													 'Cloud Coverage': clouds_x, 
													 'Temperature': temp_x, 
													 'Relative Humidity': humid_x,
													 'ETR': etr_x,
													 'DNI': dni_x,
													 'DHI': dhi_x})
	write_df.to_csv(fp)
	return write_df













