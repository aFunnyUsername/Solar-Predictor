#we're going to try adding in ALL the data now, including night time
#since the TMY data is hourly, we'll have an index from 0 to 8759 (8760 hrs in a year)
import sys
#---------------
import datetime
#---------------
import scipy
#---------------
import numpy as np
#---------------
import pandas as pd
from pandas.tools.plotting import scatter_matrix
#---------------
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LassoLars 
from sklearn.linear_model import BayesianRidge 
from sklearn.linear_model import SGDRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn import preprocessing as pp
from sklearn.externals import joblib
#===================================================================================
#===================================================================================
#===================================================================================
filename = 'tmy3MSPairport.csv'
tmy = pd.read_csv(filename, header=1, low_memory=False)
tmy_weather = tmy[['ETR (W/m^2)', 'GHI (W/m^2)', 'DNI (W/m^2)', 'DHI (W/m^2)', 
									 'TotCld (tenths)', 'Dry-bulb (C)', 'RHum (%)']]

ETR_a = np.array(tmy_weather['ETR (W/m^2)'].values, dtype=np.float)
GHI_a = np.array(tmy_weather['GHI (W/m^2)'].values, dtype=np.float)
DNI_a = np.array(tmy_weather['DNI (W/m^2)'].values, dtype=np.float)
DHI_a = np.array(tmy_weather['DHI (W/m^2)'].values, dtype=np.float)
TotClds_a = np.array(tmy_weather['TotCld (tenths)'].values, dtype=np.float)
DryBulb_a = np.array(tmy_weather['Dry-bulb (C)'].values, dtype=np.float)
RHum_a = np.array(tmy_weather['RHum (%)'].values, dtype=np.float)


#NOTE we're actually scaling here using scikit's preprocessing (min max is normalizing)
tmy_weather_a = tmy_weather.values

def normalizer(array):
	normalizer = pp.MinMaxScaler()
	return normalizer.fit_transform(array)	
def inverse(array):
	normalizer = pp.MinMaxScaler()
	return normalizer.inverse_transform(array)

X = tmy_weather_a[:, [5, 6, 0, 4]]
Y = tmy_weather_a[:, [2, 3]]
DNI_Y = Y[:, 0]
DHI_Y = Y[:, 1]

validation_size = 0.20
DNI_seed = 7
DHI_seed = 5
X_train, X_validation, DNI_Y_train, DNI_Y_validation = model_selection.train_test_split(X, DNI_Y, test_size=validation_size, random_state=DNI_seed) 
X_train, X_validation, DHI_Y_train, DHI_Y_validation = model_selection.train_test_split(X, DHI_Y, test_size=validation_size, random_state=DNI_seed)

#NOTE WE'RE ACTUALLY SCALING HERE BECAUSE WE NEED THE VALIDATION SCALER TO BE ITS OWN THING!!!!!  
#need to reshape the Y arrays I think:
DNI_Y_train = np.array(DNI_Y_train).reshape((len(DNI_Y_train), 1))
DHI_Y_train = np.array(DHI_Y_train).reshape((len(DHI_Y_train), 1))
DNI_Y_validation = np.array(DNI_Y_validation).reshape((len(DNI_Y_validation), 1))
DHI_Y_validation = np.array(DHI_Y_validation).reshape((len(DHI_Y_validation), 1))

X_normalizer = pp.MinMaxScaler()
DNI_Y_normalizer = pp.MinMaxScaler()
DHI_Y_normalizer = pp.MinMaxScaler()

X_train_scaled = X_normalizer.fit_transform(X_train)
DNI_Y_train_scaled = DNI_Y_normalizer.fit_transform(DNI_Y_train)
DHI_Y_train_scaled = DHI_Y_normalizer.fit_transform(DHI_Y_train)

scores = ['neg_mean_absolute_error', 'r2']

models = []
#build a list of models that we will iterate through
#NOTE WE'RE TRYING REGRESSION ALGORTIHMS THIS TIME INSTEAD
#models.append(('SGD', SGDRegressor()))
#models.append(('BR', BayesianRidge()))
#models.append(('LL', LassoLars()))
#models.append(('ARD', ARDRegression()))
#models.append(('PA', PassiveAggressiveRegressor()))
#models.append(('TS', TheilSenRegressor()))
models.append(('SVM', SVR()))

DNI_r2_mean = []
DNI_r2_std = []
DNI_MAE_mean = []
DNI_MAE_std = []
DHI_r2_mean = []
DHI_r2_std = []
DHI_MAE_mean = []
DHI_MAE_std = []
model_names = []




for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=DNI_seed)
	model_names.append(name)
	for method in scores:
		cv_results = model_selection.cross_val_score(model, X_train_scaled, DNI_Y_train_scaled.ravel(), cv=kfold, scoring=scores[scores.index(method)])
		if method == 'neg_mean_absolute_error':
			DNI_MAE_mean.append(cv_results.mean())
			DNI_MAE_std.append(cv_results.std())	
		elif method == 'r2':
			DNI_r2_mean.append(cv_results.mean())	
			DNI_r2_std.append(cv_results.std())	
		print('score added')
	print('model added')

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=DHI_seed)
	#model_names.append(name)
	for method in scores:
		cv_results = model_selection.cross_val_score(model, X_train_scaled, DHI_Y_train_scaled.ravel(), cv=kfold, scoring=scores[scores.index(method)])
		if method == 'neg_mean_absolute_error':
			DHI_MAE_mean.append(cv_results.mean())
			DHI_MAE_std.append(cv_results.std())	
		elif method == 'r2':
			DHI_r2_mean.append(cv_results.mean())	
			DHI_r2_std.append(cv_results.std())	
		print('score added')
	print('model added')

DNI_results_df = pd.DataFrame({
	'R Squared mean: ': DNI_r2_mean,
	'R Squared std: ': DNI_r2_std,
	'Mean Absolute Value mean: ': DNI_MAE_mean,
	'Mean Absolute Value std: ': DNI_MAE_std,
	}, 
	index=model_names)

DHI_results_df = pd.DataFrame({
	'R Squared mean: ': DHI_r2_mean,
	'R Squared std: ': DHI_r2_std,
	'Mean Absolute Value mean: ': DHI_MAE_mean,
	'Mean Absolute Value std: ': DHI_MAE_std,
	}, 
	index=model_names)
print("DNI: ")
print(DNI_results_df)
print("DHI: ")
print(DHI_results_df)



"""
DNI results:
     Mean Absolute Value mean:   Mean Absolute Value std:   R Squared mean:   
SGD                   -0.136732                   0.003001          0.604588
BR                    -0.136478                   0.003515          0.614783
LL                    -0.226957                   0.005188         -0.002201
PA                    -0.301879                   0.273329         -1.717961
TS                    -0.131138                   0.003943          0.558442

     R Squared std:
SGD         0.020322
BR          0.022353
LL          0.001984
PA          4.202320
TS          0.013613
DHI results:
     Mean Absolute Value mean:   Mean Absolute Value std:   R Squared mean:   
SGD                   -0.057182                   0.001963          0.829254
BR                    -0.055235                   0.001988          0.839168
LL                    -0.160193                   0.003813         -0.001030
PA                    -0.087314                   0.028066          0.734309
TS                    -0.047079                   0.002315          0.823808

     R Squared std:
SGD         0.009882
BR          0.010986
LL          0.000915
PA          0.132762
"""



"""
DNI results:
     Mean Absolute Value mean:   Mean Absolute Value std:   R Squared mean:   
SVM                   -0.076914                   0.004595          0.833813

     R Squared std:
SVM         0.028389
DHI results:
     Mean Absolute Value mean:   Mean Absolute Value std:   R Squared mean:   
SVM                    -0.05143                   0.002682          0.876726

     R Squared std:
SVM         0.013468
"""



#Looks like the regression model of the Support Vector Machine is our best model on the test data
X_validation_scaled = X_normalizer.fit_transform(X_validation)
DNI_Y_validation_scaled = DNI_Y_normalizer.fit_transform(DNI_Y_validation)
DHI_Y_validation_scaled = DHI_Y_normalizer.fit_transform(DHI_Y_validation)

SVM_DNI = SVR()
SVM_DHI = SVR()
SVM_DNI.fit(X_validation_scaled, DNI_Y_validation_scaled.ravel())
SVM_DHI.fit(X_validation_scaled, DHI_Y_validation_scaled.ravel())

SVM_DNI_filename = 'SVR_DNI_final_06272018.sav'
SVM_DHI_filename = 'SVR_DHI_final_06272018.sav'
joblib.dump(SVM_DNI, SVM_DNI_filename)
joblib.dump(SVM_DHI, SVM_DHI_filename)

predictions_DNI = SVM_DNI.predict(X_validation_scaled)
predictions_DHI = SVM_DHI.predict(X_validation_scaled)

predictions_DNI = np.array(predictions_DNI).reshape((len(predictions_DNI), 1))
predictions_DHI = np.array(predictions_DHI).reshape((len(predictions_DHI), 1))

predictions_DNI_inverted = DNI_Y_normalizer.inverse_transform(predictions_DNI)
predictions_DHI_inverted = DHI_Y_normalizer.inverse_transform(predictions_DHI)
X_validation_inverted = X_normalizer.fit_transform(X_validation_scaled)
#DNI_Y_validation = DNI_Y_normalizer.fit_transform(DNI_Y_validation_scaled)
#DHI_Y_validation = DHI_Y_normalizer.fit_transform(DHI_Y_validation_scaled)

"""predictions_df = pd.DataFrame({
	'DNI: ' : predictions_DNI_inverted,
	'DHI: ' : predictions_DHI_inverted
})"""
#print(predictions_df)
print(mean_absolute_error(DNI_Y_validation, predictions_DNI_inverted))
print(DNI_Y_validation.mean(), predictions_DNI_inverted.mean())
print(mean_absolute_error(DHI_Y_validation, predictions_DHI_inverted))
print(DHI_Y_validation.mean(), predictions_DHI_inverted.mean())

print(X_validation_inverted)
print(predictions_DNI)
print(predictions_DHI)


#testing out the predictions on some real future data!
csv_filename = 'data_csv_nws.csv'
new_df = pd.read_csv(csv_filename, low_memory=False)

new_date_time_list = []

def remove_year(date):
	if date[0] is '0':
		new_date = date[1:-5]
	else:
		new_date = date[:-5]
	return new_date

def remove_zero_time(time):
	if time[0] is '0':
		new_time = time[1:]
	else:
		new_time = time
	return new_time

j=0
for time in new_df['Date']:
	combined_dt = remove_year(new_df.loc[j, 'Date']) + ' ' + remove_zero_time(new_df.loc[j, 'Time'])
	new_date_time_list.append(combined_dt)
	j += 1

new_date_time_df = pd.DataFrame({'DateTime': new_date_time_list})
#print(new_date_time_df)
#print(new_df)

tmy_date_time_list = []

i = 0
for row in tmy['Date (MM/DD/YYYY)']:
	combined_dt = remove_year(tmy.loc[i, 'Date (MM/DD/YYYY)']) + ' ' + remove_zero_time(tmy.loc[i, 'Time (HH:MM)'])
	tmy_date_time_list.append(combined_dt)
	i += 1

tmy_date_time_df = pd.DataFrame({'DateTime': tmy_date_time_list, 'ETR': ETR_a})


n = 0
future_ETR = []
for index in tmy_date_time_df['DateTime'].isin(new_date_time_df['DateTime']):
	if index:
		future_ETR.append(tmy_date_time_df.loc[n, 'ETR'])
	n += 1


new_df['ETR'] = future_ETR

new_df_a = new_df.values
new_X = new_df_a[:, [4, 5, 6, 3]]


new_X_scaled = X_normalizer.fit_transform(new_X)


future_DNI = SVM_DNI.predict(new_X_scaled)
future_DHI = SVM_DHI.predict(new_X_scaled)

future_DNI = np.array(future_DNI).reshape((len(future_DNI), 1))
future_DHI = np.array(future_DHI).reshape((len(future_DHI), 1))

future_DNI_inverted = DNI_Y_normalizer.inverse_transform(future_DNI)
future_DHI_inverted = DHI_Y_normalizer.inverse_transform(future_DHI)

new_df['DNI'] = future_DNI_inverted
new_df['DHI'] = future_DHI_inverted

print(new_df)











