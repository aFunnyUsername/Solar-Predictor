# Solar-Predictor

The beginnings of a project to use data analysis and machine learning techniques to predict the output of a solar energy generation system.

## Concept

The current state of the US electric grid does not provide a good fit for the variable behavior of solar (and other renewable) generation sources
Some modelling of systems exists and has been used to some extent by utilities looking to predict their load more effectively given the new generation sources.
However, the models that I have encountered give total monthly kilowatt hour generation numbers at their most granular.  The reason for this being that the financial relationship
between the utility and developer is a Power Purchase Agreement, meaning that all killowatt hours generated will be purchased by the utility for a certain price.

However, in order to truly facilitate the ever increasing injection of solar generation into the electric grid, more real-time prediction will be necessary, hopefully down to a few hours out.

I wanted to see what could be done to assist in this area using free or inexpensive resources.  Links will be at the bottom of the readme.

## Getting Started

The general structure of the program is this.  The runner.py script sets up an hourly schedule to:

1. Pull an xml file from the National Digital Forecast Database (NDFD) (ndfdAPI.py)
2. Parse through the xml file and save the data to a csv (readXML.py)
3. Make predictions based on this data with a machine learning algorithm (currently Support Vector Regression) and store each of these hourly predictions in a csv (predictor.py).  

A couple of notes on the general layout:

1. The NDFD, according to their website, is updated every hour.  Note however, that the data is stored every three hours: 0100, 0400, 0700, 1000, 1300, 1600, 1900, 2200, repeated.  The reader pulls every hour in case a change has been made however,
2. In order for a user to use the program, the runner.py script will need to be updated: the filepaths on lines 22-29 and 56 will all need to be updated for the user's system.  In addition, you can update your Other than that, I think all other scripts should work properly - please let me know if this is not the case and I will work to resolve it.

## Data Analysis

Note, there will be more to come in this section, but for the time being, I want to lay out the general idea of how the system will make a prediction.

1. The model is actually not being trained on NDFD data but on TMY3 weather data from the National Renewable Energy Laboratory (NREL).  NREL has various locations throughout the country.  In this case, we are using data from the Minneapolis-St. Paul airport in Bloomington, MN, however, I am planning on studying various areas throughout the country.
2. The attributes I am currently interested in are as follows: Total Cloud Cover (% of the sky dome), Relative Humidity, Temperature and Extra Terrestrial Irradiance (ETR).  I am attempting to predict the Direct Normal Irradiance and the Diffuse Horizontal Irradiance.  I will break each of these down below:

### Total Cloud Cover
As was stated above, this is the percentage of the sky-dome that is covered in clouds.  Note that TMY3 data gives this value in "tenths" of the sky dome (0 being 0% and 10 being 100%, 5 would be 50%, etc.).  NDFD gives the value in a percentage, which I have divided by 10 in order to bring it to the same scale when making predictions.  However, I feel as though I need to remove the extra significant figure from the prediction data since it does not exist in the training data.  Cloud Cover (outside of ETR) seems to be the strongest indicator of solar irradiance at a given time, as they are strongly negatively correlated (r is about -.7 or -.8).

### Relative Humidity
At first I was tempted to think that cloud cover would be the only significant factor in determining the total solar irradiance at a location however, I came to the conclusion that this is not the case.  First of all, which I will demonstrate more mathematically later, cloud cover, relative humidity and temperature are all correlated to some degree, meaning that the latter two attributes contribute to the total cloud cover and thus the solar irradiance.  This makes sense, if one considers that an 80% covered sky dome on a dry day may have much thinner clouds than that of an 80% covered sky dome on a humid day.  

### Temperature
Note, NDFD gives in Fahrenheit which I am converting to Celsius at the moment since TMY gives in Celsius.  However, there is a command that can be sent in the payload to the NDFD server to indicate we want metric units so I will likely make that change later.

### Other Attributes to Consider Later
Ceiling Height would be interesting to consider at some point, however, NDFD doesn't have a value for this directly, so it would need to be thought through more.


