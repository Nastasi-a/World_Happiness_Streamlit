import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

df=pd.read_csv("merged_happiness_dataframe.csv")

st.title("World Happiness Report : regression project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)


#Creation of Exploration page

if page == pages[0] : 
  st.write("### Presentation of data")

st.dataframe(df.head(10))

st.write(df.shape)
st.dataframe(df.describe())

if st.checkbox("Show NA") :
  st.dataframe(df.isna().sum())


#Creation of Data Vizualization page

if page == pages[1] : 
  st.write("### DataVizualization")

#Bar plot to show the distribution of the Ladder score in 2021.

df_2021 = df[df['year'] == 2021]

fig = px.bar(df_2021, x='Country name', y='Ladder score')
fig.update_layout(title='Ladder Score in 2021 by Country')
st.plotly_chart(fig)

#Comparison bar plots between countries with highest and lowest 'Ladder score' in 2021.

Top10_2021 = df_2021.sort_values(by = 'Ladder score', ascending = False).head(10)
Last10_2021 = df_2021.sort_values(by = 'Ladder score', ascending = True).head(10)


from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Top 10 Ladder Score 2021', 'Last 10 Ladder Score 2021'])

fig.add_trace(px.bar(Top10_2021, x='Country name', y='Ladder score').data[0], row=1, col=1)

fig.add_trace(px.bar(Last10_2021, x='Country name', y='Ladder score').data[0], row=1, col=2)

yticks_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
fig.update_yaxes(tickvals=yticks_values)

st.plotly_chart(fig)

#Creating 'Ladder score category' variable in the 2021 dataframe with the values 'ladder score low', 'ladder score medium' and 'ladder score high' with the help of the quantiles.

def category(ladder_score, q1, q3):
    if ladder_score < q1:
        return 'ladder score low'
    elif q1 <= ladder_score <= q3:
        return 'ladder score medium'
    else:
        return 'ladder score high'

q1 = df_2021['Ladder score'].quantile(0.25)
q3 = df_2021['Ladder score'].quantile(0.75)

df_2021['Ladder score category'] = df_2021['Ladder score'].apply(lambda x: category(x, q1, q3))

import geopandas as gpd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) #Getting country data from geopandas.
world = world.rename({'name':'Country name'}, axis =1) #Renaming column to match our dataframe.
world.loc[world['Country name'] == 'United States of America', 'Country name'] = 'United States' #renaming country name to match our dataset

#Create world map showing the 'Ladder score categories'

world_merged = world.merge(df_2021, left_on='Country name', right_on='Country name')

gdf = gpd.GeoDataFrame(world_merged)

# Map categories to colors
color_dict = {'ladder score high': 'green', 'ladder score medium': 'yellow', 'ladder score low': 'red'}

# Map category colors to the GeoDataFrame
gdf['color'] = gdf['Ladder score category'].map(color_dict)

# Create a plot
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
gdf.plot(color=gdf['color'], linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

# Add legend
ax.legend()

# Set title
ax.set_title('World Happiness Score Categories low / medium / high')

# Display the plot using Streamlit
st.pyplot(fig)


#Creation of Modelling page

if page == pages[2] : 
  st.write("### Modelling")

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

feats = df.drop(['Ladder score'], axis=1)
target = df[['Ladder score']]

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42) #Splitting the data on the train and test sets

cat = ['Regional indicator', 'Country name']
oneh = OneHotEncoder(drop = 'first', sparse_output = False, handle_unknown = 'ignore')

X_train_encoded = oneh.fit_transform(X_train[cat])
X_test_encoded = oneh.transform(X_test[cat])

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=oneh.get_feature_names_out(cat)) #Creating of a new dataframe
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=oneh.get_feature_names_out(cat)) #Creating of a new dataframe

X_train.reset_index(drop=True, inplace=True) #Resetting of indices to avoid not matching indices while concatenation
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train = pd.concat([X_train.drop(columns=cat), X_train_encoded_df], axis=1) #Concatenate encoded categorical features with other features in X_test
X_test = pd.concat([X_test.drop(columns=cat), X_test_encoded_df], axis=1)#Concatenate encoded categorical features with other features in X_train

#Scaling numerical variables.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load the trained linear regression model
lr = joblib.load('trained_lr.joblib')

lr_predictions = lr.predict(X_test)

# Display R² score on the train set
st.write('Score on the train set with Linear Regression:', lr.score(X_train, y_train))

# Display R² score on the test set
st.write('Score on the test set with Linear Regression:', lr.score(X_test, y_test))
#comment

