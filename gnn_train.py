#Alireza Bordbar 
# bordbar@chalmers.se

# import the dependencies 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import random 
import torch 
import string 
import re 

import os 


#read the dataframe from Kaggle 
movies_df = pd.read_csv("data/movies.csv",index_col='movieId')
ratings_df = pd.read_csv("data/ratings.csv")

print("The dataframes were imported successfully!")


#The dataset needs a bit of cleaning up before feeding it into the network. 

