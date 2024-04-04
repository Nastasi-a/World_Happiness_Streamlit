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

#I added the comment
