#This will be an example of walking through a univariate time series problem
#with the TMY3 dataset while following the walkthrough we did earlier in the
#week.  I will attempt to write "good code", meaning I will use the 
#utilities.py program and update it with more functions as it seems useful to
#do so.  Leggo.

#We will just be looking at DNI and DHI for this exercise, but each one
#separately.

import utilities_ts as util 
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

values_dni = ['DNI (W/m^2)']
values_dhi = ['DHI (W/m^2)']
dni_df = util.parse_index_time_column('tmy3MSPairport.csv', values_dni)
dhi_df = util.parse_index_time_column('tmy3MSPairport.csv', values_dhi)
#ok, so now we have the series that we will be working with for the univariate
#reduced problem.
#first, we need to make a test harness.  This will involve two steps:
#1. Define a validation set
#2. Develop a method for model evaluation.

#1. Define a validation set.
#I think that holding back the last 2 months of the year for validation would 
#be a good starting point.  This would be the final 1464 hours.
validation = 1464
dni_dataset, dni_validation = util.split_dataset_validation(dni_df, validation)
dhi_dataset, dhi_validation = util.split_dataset_validation(dhi_df, validation)

#2. Develop a method for model evaluation.
#We need to come up with two things for this:
#1. Performance Measure
#2. Test Strategy

#Performance Measure:
#I think RMSE would be a good measure of performance here.  I expect that the
#daytime predictions will be much better than nighttime/morning/evening 
#predictions and I want the performance measure to display this.
#This is from the observation of a stronger correlation as the irradiance 
#increased.  I will write this code later.

#Test Strategy:
#Of the remaining 10 months, we will use walk-forward validation.  This is
#because a rolling-forecast type model is required from the problem definition.
#This is where one-step forecasts are needed given all available data.
#1. The first 50% of the remaining 10 months (January-May) will be used to
#train the model.
#2. The remaining 50% of the remaining 10 months (July-October) will be 
#iterated and test the model.
#3. For each step in the test dataset:
#	a) A model will be trained.
# b) A one-step prediction made and the prediction stored for later evaluation.
# c) The actual observation from the test dataset will be added to the training
#dataset for the next iteration.
#NOTE, this is our train/test split.  The split before was for validation.
#NOTE, we should explore different data configurations for this problem.
#4. The predictions made during the iteration of the test dataset will be 
#evaluated and an RMSE score reported.
split = 0.50
dni_train, dni_test = util.split_dataset_train_test(
	dni_dataset, values_dni, split)
dhi_train, dhi_test = util.split_dataset_train_test(
	dhi_dataset, values_dhi, split)
#Next, we can iterate over the time steps in the test dataset.  The train 
#dataset is stored in a python list as we need to easily append a new 
#observation each iteration and NumPy array concatenation feels like overkill.
#NOTE, the array concatenation may very well be necessary as more variables are
#added.
#This will get coded later


#OK So first we need to develop a baseline of performance.  The baseline we 
#will be using is called the naive forecast, or persistence.  We use the
#observation at the previous time step as a prediction for the observation
#at the next step. 

dni_baseline_rmse = util.naive_baseline(dni_train, dni_test)
dhi_baseline_rmse = util.naive_baseline(dhi_train, dhi_test)
#NOTE, this gives rmse values of 134.09 and 45.70 respectfully.

#Now, we'll so some data analysis.
#Use summary statistics and plots of the data to quickly learn more about the
#structure of the prediction problem.  
#1. Summary Stats
#2. Line Plot
#3. Density Plot
#4. Box and Whisker Plot

#So, for summary stats, the mean, std don't mean much because it's an 
#exponential distribution.  We'll have to do a transform before they do.

















