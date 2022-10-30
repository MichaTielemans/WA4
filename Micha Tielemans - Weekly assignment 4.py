#!/usr/bin/env python
# coding: utf-8

# In this assignment, you are going to apply what you learned about machine learning to a dataset of your choice on KaggleLinks to an external site.. Kaggle is an online platform for data science. Predict the outcomes in a data set using either Random Forest or k-NN.
# 
# The documentation is even more important than the code. Explain what you are doing and why. Only comments on the code should be in coding formatting.

# # Introduction

# What influences love at first sight? (Or, at least, love in the first four minutes?) This dataset was compiled by Columbia Business School professors Ray Fisman and Sheena Iyengar for their paper Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment.
# 
# Data was gathered from participants in experimental speed dating events from 2002-2004. During the events, the attendees would have a four minute "first date" with every other participant of the opposite sex. At the end of their four minutes, participants were asked if they would like to see their date again. They were also asked to rate their date on six attributes: Attractiveness, Sincerity, Intelligence, Fun, Ambition, and Shared Interests.
# 
# The dataset also includes questionnaire data gathered from participants at different points in the process. These fields include: demographics, dating habits, self-perception across key attributes, beliefs on what others find valuable in a mate, and lifestyle information. See the Speed Dating Data Key document below for details.
# 
# We are going to predict the variable dec_o (decision by partner) using this dataset, to see what influences a decisionmaking process of the dates. I choose this dataset, because i was interested in what could be influential for a succesful date. 

# In[2]:


import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Let's first look at the dataset and see which variables we can use.

# In[13]:


SD_data = pd.read_csv('Speed Dating Data.csv')
SD_data.head(20)


# - We don't need to use any of the ID value's
# - dec_o is our dependent variable
# - There are 6 other valuables i want to use for this model: field_cd, mn_satm, zipcode, income, goal and date. This is because I felt this would give me the most interesting results in prediction the dec_o variable. 
# 
# So let's select those variables and drop the rows with NaN's so we can use the information later on.

# # Data cleaning

# In[14]:


df = SD_data[['dec_o','field_cd', 'mn_sat', 'zipcode', 'income', 'goal', 'date']]
df = df.dropna() #get rid of rows with empty cells
df.head(20)


# I don't think that we need to look for impossible values, since all of the data is from a survey and has been categorised. 
# 
# Let's see how many times the decision of partner the night of event was positive. We can use this for the evaluation.

# In[15]:


df['dec_o'].value_counts()


# # Exploratory data analysis

# Let's have a look at the correlation between the variables

# In[20]:


SD_data.corr().loc['dec_o'].sort_values(ascending=False).head()


# # Predictive model (next week)

# x

# # Evaluation (next week)

# x

# # Conclusion (next week)

# x

# In[ ]:




