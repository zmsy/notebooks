# coding: utf-8

# # AutoML Runs Created
#
# This is a model to try to predict how many runs created a specific player will get over the next few weeks.

# In[2]:


import autosklearn as ask
import autosklearn.regression as rgr
from sklearn import model_selection
import pandas as pd
import numpy as np
import os
import sqlalchemy
import psycopg2
import pybaseball
import seaborn as sb
import requests
import datetime
import pandas_datareader as pdr

# arbitrarily high max view columns
pd.options.display.max_columns = 300
pd.options.display.max_rows = 200


# In[3]:


# get data from fangraphs
df = pybaseball.batting_stats("2019")


# In[4]:


# choose list of valid columns to keep
df.columns = [x.lower() for x in df.columns]
columns_to_keep = [
    "age",
    "g",
    "ab",
    "pa",
    "h",
    "1b",
    "2b",
    "3b",
    "hr",
    "r",
    "rbi",
    "bb",
    "ibb",
    "so",
    "hbp",
    "sf",
    "sh",
    "gdp",
    "sb",
    "cs",
    "avg",
    "gb",
    "fb",
    "ld",
    "iffb",
    "pitches",
    "balls",
    "strikes",
    "ifh",
    "bu",
    "buh",
    "bb%",
    "k%",
    "bb/k",
    "obp",
    "slg",
    "ops",
    "iso",
    "babip",
    "gb/fb",
    "ld%",
    "gb%",
    "fb%",
    "iffb%",
    "hr/fb",
    "ifh%",
    "buh%",
    "woba",
    "wraa",
    "bat",
    "rep",
    "pos",
    "rar",
    "war",
    "spd",
    "wrc+",
    "wpa",
    "-wpa",
    "+wpa",
    "re24",
    "rew",
    "pli",
    "ph",
    "wpa/li",
    "fbv",
    "wfb",
    "wsl",
    "wct",
    "wcb",
    "wfb/c",
    "wsl/c",
    "wct/c",
    "wcb/c",
    "o-swing%",
    "z-swing%",
    "swing%",
    "o-contact%",
    "z-contact%",
    "contact%",
    "zone%",
    "f-strike%",
    "swstr%",
    "bsr",
    "pace",
    "def",
    "wsb",
    "ubr",
    "off",
    "lg",
    "wgdp",
    "pull%",
    "cent%",
    "oppo%",
    "soft%",
    "med%",
    "hard%",
    "tto%",
]
df = df[columns_to_keep]
df_y = df["ab"] * df["obp"] * df["slg"]


# In[5]:


regressor = rgr.AutoSklearnRegressor()
x_train, x_test, y_train, y_test = model_selection.train_test_split(df, df_y)


# In[1]:


regressor.fit(x_train, y_train, dataset_name="boston")
