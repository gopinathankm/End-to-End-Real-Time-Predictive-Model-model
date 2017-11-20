#--------------------------------------------------------------------------------------
# Imports
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

import numpy as np
#------------------------------------------------------------------------------------
# Load the data 
power_plant = pd.read_excel("./linear-regression/CCPP/Folds5x2_pp.xlsx")

type(power_plant)

# See first few rows
power_plant.head(5)

print(power_plant.shape)

print(power_plant.columns.values)

# class of each column in the DataFrame
power_plant.dtypes

print(power_plant.describe())

# Are there any missing values in any of the columns?
power_plant.info()  # There is no missing data in all of the columns

#---------------------------------------------------------------------------
# Correlation between power output and temperature
power_plant.plot(x ='AT', y = 'PE', kind ="scatter", figsize = [10,10], color ="b", alpha = 0.3, fontsize = 14)

plt.title("Temperature vs Power Output", fontsize = 24, color="darkred")

plt.xlabel("Atmospheric Temperature", fontsize = 18) 

plt.ylabel("Power Output", fontsize = 18)

plt.show()

#----------------------------------------------------------------------------
# Correlation between Exhaust Vacuum Speed and power output 

power_plant.plot(x ='V', y = 'PE',kind ="scatter", 
                 figsize = [10,10],
                 color ="g", alpha = 0.3, 
                fontsize = 14)

plt.title("Exhaust Vacuum Speed vs Power Output", fontsize = 24, color="darkred")

plt.xlabel("Atmospheric Temperature", fontsize = 18) 

plt.ylabel("Power Output", fontsize = 18)

plt.show()

#------------------------------------------------------------------------------
# Correlation between power output and atmospheric pressure

power_plant.plot(x ='AP', y = 'PE',kind ="scatter", 
                 figsize = [10,10],
                 color ="r", alpha = 0.3,
                fontsize = 14)

plt.title("Atmospheric Pressure vs Power Output", fontsize = 24, color="darkred")

plt.xlabel("Atmospheric Pressure", fontsize = 18) 

plt.ylabel("Power Output", fontsize = 18)

plt.show()
#--------------------------------------------------------------------------------
# Correlation between relative humidity  and power output 

power_plant.plot(x ='RH', y = 'PE',kind ="scatter", 
                 figsize = [10,10],
                 color ="m", alpha = 0.3)

plt.title("Relative Humidity vs Power Output", fontsize = 24, color="darkred")

plt.xlabel("Relative Humidity", fontsize = 18) 

plt.ylabel("Power Output", fontsize = 18)

plt.show()  
#---------------------------------------------------------------------------------
# correlation heatmap 

corr = power_plant.corr()
plt.figure(figsize = (9, 7))
sns.heatmap(corr, cmap="RdBu",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
#-----------------------------------------------------------------------------------
# Splitting dataset into training and test  dataset
X = power_plant.drop("PE", axis = 1).values
y = power_plant['PE'].values
y = y.reshape(-1, 1)

# Split into training and test set
# 80% of the input for training and 20% for testing

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                               test_size = 0.2, 
                                               random_state=42)

Training_to_original_ratio = round(X_train.shape[0]/(power_plant.shape[0]), 2) * 100

Testing_to_original_ratio = round(X_test.shape[0]/(power_plant.shape[0]), 2) * 100

print ('As shown below {}% of the data is for training and the rest {}% is for testing.'.format(Training_to_original_ratio, 
                                                                                               Testing_to_original_ratio))
list(zip(["Training set", "Testing set"],
   [Training_to_original_ratio, Testing_to_original_ratio]))

#--------------------------------------------------------------------------------------
# Linear Regression
# Instantiate linear regression: reg
# Standardize features by removing the mean 
# and scaling to unit variance using the
# StandardScaler() function

# Apply Scaling to X_train and X_test

std_scale = StandardScaler().fit(X_train)
X_train_scaled = std_scale.transform(X_train)
X_test_scaled = std_scale.transform(X_test)

linear_reg = LinearRegression()
reg_scaled = linear_reg.fit(X_train_scaled, y_train)
y_train_scaled_fit = reg_scaled.predict(X_train_scaled)

print("R-squared for training dataset:{}".
      format(np.round(reg_scaled.score(X_train_scaled, y_train),
                      2)))

print("Root mean square error: {}".
      format(np.round(np.sqrt(mean_squared_error(y_train, 
                                        y_train_scaled_fit)), 2)))

coefficients = reg_scaled.coef_
features = list(power_plant.drop("PE", axis = 1).columns)

print(" ")
print('The coefficients of the features from the linear model:')
print(dict(zip(features, coefficients[0])))

print("")

print("The intercept is {}".format(np.round(reg_scaled.intercept_[0],3)))
#---------------------------------------------------------------------------------------------
# Predict using the test data

pred = reg_scaled.predict(X_test_scaled)

print("R-squared for test dataset:{}".
      format(np.round(reg_scaled.score(X_test_scaled, 
                                       y_test),  2)))


print("Root mean square error for test dataset: {}".
      format(np.round(np.sqrt(mean_squared_error(y_test, 
                                        pred)), 2)))


data =  {"prediction": pred, "observed": y_test}
print("Predictions:.....................................")
print(pred)
print("Observers........................................")
print(y_test)
print("data---------------------------------------------")
print(data)
test = pd.DataFrame(pred, columns = ["Prediction"])

test["Observed"] = y_test


lowess = sm.nonparametric.lowess

z = lowess(pred.flatten(), y_test.flatten())


test.plot(figsize = [10,10],
          x ="Prediction", y = "Observed", kind = "scatter", color = 'darkred')

plt.title("Linear Regression: Prediction Vs Test Data", fontsize = 24, color = "darkgreen")

plt.xlabel("Predicted Power Output", fontsize = 18) 

plt.ylabel("Observed Power Output", fontsize = 18)

plt.plot(z[:,0], z[:,1], color = "blue", lw= 3)

plt.show()
#-------------------------------------------------------------------------------------------------
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.decoration import ContinuousDomain

ccpp_mapper = DataFrameMapper([
    (["AT", "V", "AP", "RH"], [ContinuousDomain(), StandardScaler()])
])


from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

ccpp_pca = PCA(n_components = 3)
ccpp_selector = SelectKBest(k = 2)


from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.pipeline import make_pipeline

ccpp_classifier = linear_model.LinearRegression()

from sklearn2pmml import PMMLPipeline

ccpp_pipeline = PMMLPipeline([
    ("mapper", ccpp_mapper),
    ("pca", ccpp_pca),
    ("selector", ccpp_selector),
    ("estimator", ccpp_classifier)
])

ccpp_pipeline.fit(power_plant, power_plant["PE"])

from sklearn.externals import joblib

joblib.dump(ccpp_pipeline, "ccpp.pkl.z", compress = 9)

