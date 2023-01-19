# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:14:30 2023

@author: sande
"""

#importing pandas module to read the file(data set) and to calculate the statistical property(Describe)
import pandas as pd

#importing numpy module to calculate the statistical properties (mean and standard deviation)
import numpy as np

#importing pyplot from matplotlib module to plot the visualization graphs
import matplotlib.pyplot as plt

#importing KMeans from sklearn object to identify the clusters
from sklearn.cluster import KMeans

#importing LabelEncoder to encode the categories of the data
from sklearn.preprocessing import LabelEncoder

#importing the custom error package 
import err_ranges as error

#importing curve_fit from scipy module to predict the population groth of Australia
from scipy.optimize import curve_fit

#defining a function to read the dataset and to produce original and transposed dataframes
def read_data_file(input_file_name,countries):
    #reading the data set using pandas module
    dataFrame = pd.read_csv(input_file_name)
    #cleaning the dataFrame by filling the NaN values with 0
    cleaned_dataFrame = dataFrame.fillna(0)
    #slicing the data frame by selecting fewe countries of our option
    sliced_dataFrame = cleaned_dataFrame[cleaned_dataFrame['Country Name'].isin(countries)]
    #creating a new data frame with countires as first column using the sliced data frame
    dataFrame_countries = pd.DataFrame(sliced_dataFrame)
    print('Original DataFrame:\n',dataFrame_countries)
    #transposing the sliced data frame
    transposed_dataFrame = pd.DataFrame.transpose(sliced_dataFrame)
    #creating a header
    header = transposed_dataFrame.iloc[0].values.tolist()
    #assigning the header to the transposed data frame
    transposed_dataFrame.columns = header
    #assigning the transposed dataframe with years as first column to a new variable
    dataFrame_years = transposed_dataFrame
    print('Transposed DataFrame:\n',dataFrame_years)
    #returning the 2 dataframes (one dataframe with countries as first column and other dataframe with years as first column)
    return dataFrame_countries,dataFrame_years

#function to calculate the logistic value
def logi(t, n0, g, t0):
    f = n0 / (1+np.exp(-g*(t - t0)))
    return f

#function to calculate the exponential value
def exponen_graph(t, n0, g):
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

#calling the function that produces two dataframes: one with countries as columns and another with years as columns
df_countries, df_years= read_data_file('C:/Users/sande/MS-DS/API_19_DS2_en_csv_v2_4700503-Copy.csv',['Afghanistan','Albania','Argentina','Austria','Belgium','Bangladesh','Brazil','Canada','Switzerland','Chile','China','Colombia','Denmark','Dominican Republic','Algeria','Spain','Finland','Fiji','France','United Kingdom','Greece','Greenland','Hungary','Indonesia','India','Ireland','Iraq','Iceland','Israel','Italy','Jamaica','Japan','Lebanon','Luxembourg','Morocco','Mexico','Myanmar','Netherlands','New Zealand','Pakistan','Peru','Poland','Romania','Russian Federation','Sweden','Thailand','Tunisia','Turkiye','Uruguay','United States','Vietnam','South Africa','Zimbabwe'])

#selecting only few idnicators for clustering and fitting purpose
selected_indicators = ['Urban population growth (annual %)','CO2 emissions from liquid fuel consumption (% of total)','CO2 emissions from solid fuel consumption (% of total)']

#filtering the original dataset with the selected indicators
selected_df = df_countries.loc[df_countries["Indicator Name"].isin(selected_indicators)]

#dropped two columns for clustering purpose
selected_df = selected_df.drop(columns=['Country Name','Indicator Name',],axis=1)

#performing data preproceesing by labelling the column as a categorical column
label_encoder = LabelEncoder()
#converting the Class column into encoding values to compare with ground truth values after clustering predictions
selected_df['Country Code'] = label_encoder.fit_transform(selected_df['Country Code'])
selected_df['Indicator Code'] = label_encoder.fit_transform(selected_df['Indicator Code'])