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
