#copy the full walkthrough, but with the TMY3 data
#1 - Import Libraries
import sys
#---------------
import scipy
#---------------
import numpy
#---------------
import matplotlib.pyplot as plt
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#-------------------------------------------------------------------
#2 - Load the Dataset, in this case its the csv we've been saving to 
filename = 'tmy3MSPairport.csv'
tmy_full_dataframe = pd.read_csv(filename, header=1, low_memory=False)
#In addition to loading in this case, there's some cleaning we want to do namely:
#A. Remove times where there is no irradiance (GHI) i.e. night time
#B. Make one data frame for DNI and other columns (clouds, etc.) and one for DHI
#We will look at these two separately

#Remove non-irradiated times:
tmy_full_rad = tmy_full_dataframe[tmy_full_dataframe['GHI (W/m^2)'] > 0]
#Split into just the relevant series: 
tmy_full_rad_relevant = tmy_full_rad[['GHI (W/m^2)', 'DNI (W/m^2)', 'DHI (W/m^2)', 'TotCld (tenths)', 'RHum (%)']] 

GHI = tmy_full_rad_relevant['GHI (W/m^2)']
DNI = tmy_full_rad_relevant['DNI (W/m^2)']
DHI = tmy_full_rad_relevant['DHI (W/m^2)']
TotClds = tmy_full_rad_relevant['TotCld (tenths)']
RHum = tmy_full_rad_relevant['RHum (%)']
#-------------------------------------------------------------------
#3 - Summarize the dataset: Dimensions, Peek at Data, Statistical Summary of all attributes

#Dimensions:
shape = tmy_full_rad_relevant.shape
#Peek:
head_20 = tmy_full_rad_relevant.head(20)
#Stats summary:
describe = tmy_full_rad_relevant.describe()

"""print(shape)
print(head_20)
print(describe)
"""
#-------------------------------------------------------------------
#4 - Visualize Data with plots

#box/whisker:
#tmy_full_rad_relevant.plot(kind='box', subplots=True, layout=(5, 5), sharex=False, sharey=False)

#histograms:
#tmy_full_rad_relevant.hist()

#scatter plots:
#scatter_matrix(tmy_full_rad_relevant, s=1)
#plt.show()
#NOTE interesting thing to look at could be the relationship between DNI/DHI and total clouds at
#different "blocks" of relative humidity (20-40%, 40-60%, etc.)

#split into "humidity blocks":
block_019 = tmy_full_rad_relevant[tmy_full_rad_relevant['RHum (%)'].between(0, 19, inclusive=True)]
block_2039 = tmy_full_rad_relevant[tmy_full_rad_relevant['RHum (%)'].between(20, 39, inclusive=True)]
block_4059 = tmy_full_rad_relevant[tmy_full_rad_relevant['RHum (%)'].between(40, 59, inclusive=True)]
block_6079 = tmy_full_rad_relevant[tmy_full_rad_relevant['RHum (%)'].between(60, 79, inclusive=True)]
block_80100 = tmy_full_rad_relevant[tmy_full_rad_relevant['RHum (%)'].between(80, 100, inclusive=True)]
"""print(block_019)
print(block_2039)
print(block_4059)
print(block_6079)
print(block_80100)
"""
s = 5 #we'll try a block size of 5 for now
"""scatter_matrix(block_019, s=s)
scatter_matrix(block_2039, s=s)
scatter_matrix(block_4059, s=s)
scatter_matrix(block_6079, s=s)
scatter_matrix(block_80100, s=s)
#plt.show()
"""
#we're going to go and try to standardize each of these
"""array_019 = block_019.values
X_019 = array_019[:, 0:block_019.shape[1] - 1]
Y_019 = array_019[:, block_019.shape[1] - 1]
print(block_019.shape[1])
print(array_019)
print(X_019)
print(Y_019)
"""
#actually, I don't think my distribution is gaussian so standardization might not be the best option
"""print(block_019.describe())
print(block_2039.describe())
print(block_4059.describe())
print(block_6079.describe())
print(block_80100.describe())
"""
#ok here we go:
def correlation(x, y):
	standard_x = (x - x.mean()) / x.std(ddof=0)
	standard_y = (y - y.mean()) / y.std(ddof=0)
	r = (standard_x * standard_y).mean()
	return(r)
#calculate's pearson's r to see correlation
#let's just look at block_4059 
GHI_019 = block_019['GHI (W/m^2)']
DNI_019 = block_019['DNI (W/m^2)']
DHI_019 = block_019['DHI (W/m^2)']
TotClds_019 = block_019['TotCld (tenths)']
RHum_019 = block_019['RHum (%)']
GHI_2039 = block_2039['GHI (W/m^2)']
DNI_2039 = block_2039['DNI (W/m^2)']
DHI_2039 = block_2039['DHI (W/m^2)']
TotClds_2039 = block_2039['TotCld (tenths)']
RHum_2039 = block_2039['RHum (%)']
GHI_4059 = block_4059['GHI (W/m^2)']
DNI_4059 = block_4059['DNI (W/m^2)']
DHI_4059 = block_4059['DHI (W/m^2)']
TotClds_4059 = block_4059['TotCld (tenths)']
RHum_4059 = block_4059['RHum (%)']
GHI_6079 = block_6079['GHI (W/m^2)']
DNI_6079 = block_6079['DNI (W/m^2)']
DHI_6079 = block_6079['DHI (W/m^2)']
TotClds_6079 = block_6079['TotCld (tenths)']
RHum_6079 = block_6079['RHum (%)']
GHI_80100 = block_80100['GHI (W/m^2)']
DNI_80100 = block_80100['DNI (W/m^2)']
DHI_80100 = block_80100['DHI (W/m^2)']
TotClds_80100 = block_80100['TotCld (tenths)']
RHum_80100 = block_80100['RHum (%)']

#look at correlation between:
#1. DNI and TotCld
#2. DHI and TotCld
#for each humidity block
#3. And how they relate to those two correlations in the non-blocked dataset
DNI_019_r = correlation(DNI_019, TotClds_019)
DHI_019_r = correlation(DHI_019, TotClds_019)
DNI_2039_r = correlation(DNI_2039, TotClds_2039)
DHI_2039_r = correlation(DHI_2039, TotClds_2039)
DNI_4059_r = correlation(DNI_4059, TotClds_4059)
DHI_4059_r = correlation(DHI_4059, TotClds_4059)
DNI_6079_r = correlation(DNI_6079, TotClds_6079)
DHI_6079_r = correlation(DHI_6079, TotClds_6079)
DNI_80100_r = correlation(DNI_80100, TotClds_80100)
DHI_80100_r = correlation(DHI_80100, TotClds_80100)
DNI_tot_r = correlation(DNI, TotClds)
DHI_tot_r = correlation(DHI, TotClds)

r_df = pd.DataFrame(
	data = { 
	'Total': [DNI_tot_r, DHI_tot_r],
	'019': [DNI_019_r, DHI_019_r],
	'2039': [DNI_2039_r, DHI_2039_r],
	'4059': [DNI_4059_r, DHI_4059_r],
	'6079': [DNI_6079_r, DHI_6079_r],
	'80100': [DNI_80100_r, DHI_80100_r]
},
	index=['DNI', 'DHI']
)
#print(r_df)
#-----------------------------------------------------------------------------------
#Ok so, what if instead of all that, we just do groupby('RHum (%)')? let's see...
tmy_full_rad_relevant_by_hum = tmy_full_rad_relevant.groupby('RHum (%)')
#NOTE each group by object maps a relative humidity value to the sum (or mean) of each of the other points
hum_sum = tmy_full_rad_relevant.groupby('RHum (%)').sum()
hum_mean = tmy_full_rad_relevant.groupby('RHum (%)').mean()
"""print(tmy_full_rad_relevant_by_hum.sum())
print(tmy_full_rad_relevant_by_hum.mean())
"""
#scatter_matrix(tmy_full_rad_relevant_by_hum.mean())
scatter_matrix(tmy_full_rad_relevant_by_hum.sum())
#plt.show()
#check r for each of those group bys 
hum_sum_DNI_r = correlation(hum_sum['DNI (W/m^2)'], hum_sum['TotCld (tenths)'])
hum_sum_DHI_r = correlation(hum_sum['DHI (W/m^2)'], hum_sum['TotCld (tenths)'])
hum_mean_DNI_r = correlation(hum_mean['DNI (W/m^2)'], hum_mean['TotCld (tenths)'])
hum_mean_DHI_r = correlation(hum_mean['DHI (W/m^2)'], hum_mean['TotCld (tenths)'])

hum_r_df = pd.DataFrame(
	data = {
	'Sum': [hum_sum_DNI_r, hum_sum_DHI_r],
	'Mean': [hum_mean_DNI_r, hum_mean_DHI_r]
	},
	index=['DNI', 'DHI']
)

#print(hum_r_df)
#NOTE let's investigate this later but move on for now
#----------------------------------------------------------------------------------
#let's just take a quick look at these again:
"""print(shape)
print(head_20)
print(describe)
"""
#5 - Evaluating Algorithms:
#In my case, I don't have a "Class" column.  I suppose, DNI or DHI will be the class column here, with all other
#attributes being what we deterimine class by.
"""#NOTE I guess I will split into two here, DNI and DHI being the "class" of each df
class_DNI = tmy_full_rad[['DNI (W/m^2)', 'TotCld (tenths)', 'RHum (%)']]
class_DHI = tmy_full_rad[['DHI (W/m^2)', 'TotCld (tenths)', 'RHum (%)']]

DNI_a = class_DNI.values
DNI_columns = len(DNI_a[0])
DHI_a = class_DHI.values
DHI_columns = len(DHI_a[0])
#NOTE what we're doing here, I think, is spltting the values into two groups:
#1. the X values, what we're using to predict, which is everything except the first column in this case
#2. the Y values, which is what we will be using to train the predictor, just the first column in this case
#NOTE we found the # of columns above 
DNI_X = DNI_a[:, 1:DNI_columns]
DNI_Y = DNI_a[:, 0]
DHI_X = DHI_a[:, 1:DHI_columns]
DHI_Y = DHI_a[:, 0]
print(DNI_X)
print(DNI_Y)
print(DHI_X)
print(DHI_Y)
"""
#NOTE ok actually, let's not split the original into 2 dataframes.  Let's just have one set of X values
#and 2 sets of Y's:
#or we could even split the two into a data array and a class array
tmy_array = tmy_full_rad_relevant.values
outputs_df = tmy_full_rad_relevant[['DNI (W/m^2)', 'DHI (W/m^2)']]
datas_df = tmy_full_rad_relevant[['TotCld (tenths)', 'RHum (%)']]

outputs_a = outputs_df.values
datas_a = datas_df.values
#now, the datas array is the only array we need for our X values
#we split the the outputs array into 2 groups for DNI and DHI respectfully

DNI_Y = outputs_a[:, 0]
DHI_Y = outputs_a[:, 1]

#now, we need to make the validation dataset
validation_size = 0.20
seed = 7 #NOTE WHAT IS SEED????
X_train, X_validation, DNI_Y_train, DNI_Y_validation = model_selection.train_test_split(datas_a, DNI_Y, test_size=validation_size, random_state=seed)
X_train, X_validation, DHI_Y_train, DHI_Y_validation = model_selection.train_test_split(datas_a, DHI_Y, test_size=validation_size, random_state=seed)
#NOTE here we set the X_train, X_validation, Y_train, Y_validation based on our validation size
#in this case, we're taking 80% of the dataset for testing, and keeping 20% for validation later

#Now we make the Test Harness
#We will use 10 fold cross validation in keeping with the tutorial, but I need to try other stuff 
#and experiment for this dataset later
"""scoring = 'accuracy' #this will get passed into a method later
#this is a ratio of correctly predicted instances divided by the total number of the instances in the df
#not the best choice in my case - we probably want something more like a percentage error from each point?
"""
#instead I will use a few different regression metrics because of the way the data is
scoring = 'neg_mean_absolute_error' 

#MAE is the sum of the absolute differences between predictions and actual values
#gives an idea of how wrong the predictions were
"""scoring = 'neg_mean_squared_error'
"""
#similar to MAE but it's the sum of the means squared.  Taking the square root of this gives
#RMSE or Root Mean Squared Error
"""scoring = 'r2'
"""
#R Squared metric provides an indication of the goodness of fit of a set of predictions to the actual values
#this is similar to pearson's r or maybe derivative of it?

#We'll build our models and use each of these scoring techniques to investigate
models = []
#build a list of models that we will iterate through
models.append(('LR', LogisticRegression()))
models.append(('LinR', LinearRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#NOTE try running with SVR (regression) as opposed to SVC (classification)

DNI_results = []
DNI_names = []
DHI_results = []
DHI_names = []
#iterate through each model and get the results blah blah
"""for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, DNI_Y_train, cv=kfold, scoring=scoring)
	DNI_results.append(cv_results)
	DNI_names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print("DNI: " + msg)
#repeat for DHI
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, DHI_Y_train, cv=kfold, scoring=scoring)
	DHI_results.append(cv_results)
	DHI_names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print("DHI: " + msg)
"""
#results look like this:
"""
r2
DNI: LR: -0.367820 (0.109631)
DNI: LinR: 0.574239 (0.035757)
DNI: KNN: -0.103691 (0.110459)
DNI: CART: -0.087951 (0.086728)
DNI: NB: 0.049051 (0.099638)
DNI: SVM: -0.090306 (0.116071)
DHI: LR: -0.975677 (0.178116)
DHI: LinR: 0.172527 (0.013497)
DHI: KNN: -0.778699 (0.169180)
DHI: CART: -0.725598 (0.111154)
DHI: NB: -1.928674 (0.362644)
DHI: SVM: -0.570290 (0.139481)

mean squared error
DNI: LR: -133830.273098 (9715.830394)
DNI: LinR: -41634.159829 (2924.346047)
DNI: KNN: -107964.588859 (9900.492598)
DNI: CART: -106649.564674 (8073.219955)
DNI: NB: -92959.793207 (8224.395287)
DNI: SVM: -106588.793207 (9965.516540)
DHI: LR: -17734.595652 (1885.481688)
DHI: LinR: -7425.036516 (420.229111)
DHI: KNN: -15936.661957 (1442.601393)
DHI: CART: -15495.726902 (1288.756447)
DHI: NB: -26128.165217 (2171.767115)
DHI: SVM: -14087.670924 (1406.462924)

absolute mean error
DNI: LR: -247.719293 (13.828187)
DNI: LinR: -157.808562 (5.014107)
DNI: KNN: -226.849728 (14.730580)
DNI: CART: -222.345380 (12.278147)
DNI: NB: -236.312228 (15.866740)
DNI: SVM: -215.303533 (15.152372)
DHI: LR: -98.139674 (6.321756)
DHI: LinR: -67.325430 (2.572286)
DHI: KNN: -94.397826 (5.381658)
DHI: CART: -92.243750 (3.922657)
DHI: NB: -123.579348 (5.695655)
DHI: SVM: -86.473641 (4.758630)
"""

#after realizing my error, it looksl ike LinR is the best method for predicting DNI (but a positive correlation)
#?
#And Either KNN or CART is best for DHI
#
linr_DNI = LinearRegression()
knn_DHI = KNeighborsClassifier()
cart_DHI = DecisionTreeClassifier()
linr_DNI.fit(X_train, DNI_Y_train)
knn_DHI.fit(X_train, DHI_Y_train)
cart_DHI.fit(X_train, DHI_Y_train)

predictions_DNI = linr_DNI.predict(X_validation)
predictions_DHI_knn = knn_DHI.predict(X_validation)
predictions_DHI_cart = cart_DHI.predict(X_validation)

"""print(mean_absolute_error(DNI_Y_validation, predictions_DNI))
print(DNI_Y_validation.mean(), predictions_DNI.mean())
print(mean_absolute_error(DHI_Y_validation, predictions_DHI_knn))
print(DHI_Y_validation.mean(), predictions_DHI_knn.mean())
print(mean_absolute_error(DHI_Y_validation, predictions_DHI_cart))
print(DHI_Y_validation.mean(), predictions_DHI_cart.mean())
"""

#read in future data taht's been collected and run this model
forecast_filename = 'data_csv_nws.csv'
forecast_data = pd.read_csv(forecast_filename, header=0, low_memory=False)
#print(forecast_data)
forecast_a = forecast_data.values
forecast_a_X = forecast_a[:, 1:3]
print(forecast_a_X)
#print(forecast_a_X)
forecast_DNI = linr_DNI.predict(forecast_a_X)
forecast_DHI = cart_DHI.predict(forecast_a_X)
forecast_data['DNI'] = forecast_DNI
forecast_data['DHI'] = forecast_DHI
print(forecast_data)







