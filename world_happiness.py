import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

df=pd.read_csv("merged_happiness_dataframe.csv")

st.title("World Happiness Report")
st.sidebar.title("Table of contents")
pages=["Framework", "Exploration", "Vizualization", "Modelling", "Interpretation", "Team"]
page=st.sidebar.radio("Go to", pages)


#Creation of Exploration page

if page == pages[1] :
  st.header("Exploration of data")

  st.dataframe(df.head(10))

  st.write(df.shape)
  st.dataframe(df.describe())

  if st.checkbox("Show NA") :
    st.dataframe(df.isna().sum())


#Creation of Data Vizualization page

if page == pages[2] : 
  st.header("Data Vizualization")


#Bar plot to show the distribution of the Ladder score in 2021.

  df_2021 = df[df['year'] == 2021]

  fig = px.bar(df_2021, x='Country name', y='Ladder score')
  fig.update_layout(title='Ladder Score in 2021 by Country')
  st.plotly_chart(fig)

#Comparison bar plots between countries with highest and lowest 'Ladder score' in 2021.

  Top10_2021 = df_2021.sort_values(by = 'Ladder score', ascending = False).head(10)
  Last10_2021 = df_2021.sort_values(by = 'Ladder score', ascending = True).head(10)


  from plotly.subplots import make_subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=['Top 10 Ladder Score 2021', 'Last 10 Ladder Score 2021'])

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

if page == pages[3] : 
  st.header("Modelling")



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
  rf = joblib.load('trained_rf.joblib')
  dt = joblib.load('trained_dt.joblib')

  lr_predictions = lr.predict(X_test)
  rf_predictions = rf.predict(X_test)
  dt_predictions = dt.predict(X_test)

# Display R² score on the train and test set

  choice = ['Linear Regression', 'Random Forest', 'Decision Tree']
  option = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :', option)

  if option == 'Linear Regression':
    st.write('Score on the train set with Linear Regression:', lr.score(X_train, y_train))
    st.write('Score on the test set with Linear Regression:', lr.score(X_test, y_test))

  elif option == 'Random Forest': 
    st.write('Score on the train set with Random Forest:', rf.score(X_train, y_train))
    st.write('Score on the test set with Random Forest:', rf.score(X_test, y_test))

  else: 
    st.write('Score on the train set with Decision Tree:', dt.score(X_train, y_train))
    st.write('Score on the test set with Decision Tree:', dt.score(X_test, y_test))

  if st.checkbox("Comparison of the three models"):
    
#Comparison bar plot of R² score of the 3 models.

    lr_train_score = lr.score(X_train, y_train)
    lr_test_score = lr.score(X_test, y_test)

    rf_train_score = rf.score(X_train, y_train)
    rf_test_score = rf.score(X_test, y_test)

    dt_train_score = dt.score(X_train, y_train)
    dt_test_score = dt.score(X_test, y_test)

    model_names = ['Linear Regression Train', 'Linear Regression Test', 
               'Random Forest Train', 'Random Forest Test', 
               'Decision Tree Train', 'Decision Tree Test']

    scores = [lr_train_score, lr_test_score, 
          rf_train_score, rf_test_score, 
          dt_train_score, dt_test_score]

    df_comparison_models = pd.DataFrame({'Model': model_names, 'Score': scores})

    fig = px.bar(df_comparison_models, x='Model', y='Score', color='Model', 
             color_discrete_map={'Linear Regression Train': 'red', 
                                 'Linear Regression Test': 'red', 
                                 'Random Forest Train': 'blue', 
                                 'Random Forest Test': 'blue', 
                                 'Decision Tree Train': 'green', 
                                 'Decision Tree Test': 'green'})

    fig.update_layout(
      title='Performance on Train and Test Sets of all three models',
      yaxis_title='R² Score',
      yaxis=dict(range=[0, 1]))

    st.plotly_chart(fig)


#Comparison Boxplot of predictions of three models 

    predictions_df = pd.DataFrame({
      'Linear Regression': lr_predictions.flatten(),
      'Random Forest': rf_predictions.flatten(),
      'Decision Tree': dt_predictions.flatten()})

    fig = px.box(predictions_df.melt(var_name='Model', value_name='Predicted Value'), 
             x='Model', y='Predicted Value', color='Model',
             color_discrete_map={'Linear Regression': 'red', 'Random Forest': 'blue', 'Decision Tree': 'green'},
             labels={'Model': 'Model', 'Predicted Value': 'Predicted Value'},
             title='Distribution of predicted values for the three models')

    fig.update_layout(height=600, width=800)

    st.plotly_chart(fig)


#Scatter plots of Predicted vs. Actual Values with Diagonal Line

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(lr_predictions, y_test, c='red', s=30, marker='o')
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Adding the diagonal line
    axs[0].set_title('Linear Regression')
    axs[0].set_xlabel('Predicted Values')
    axs[0].set_ylabel('Actual Values')

    axs[1].scatter(rf_predictions, y_test, c='blue', s=30, marker='o')
    axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Adding the  diagonal line
    axs[1].set_title('Random Forest')
    axs[1].set_xlabel('Predicted Values')
    axs[1].set_ylabel('Actual Values')

    axs[2].scatter(dt_predictions, y_test, c='green', s=30, marker='o')
    axs[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Adding the diagonal line
    axs[2].set_title('Decision Tree')
    axs[2].set_xlabel('Predicted Values')
    axs[2].set_ylabel('Actual Values')

    plt.subplots_adjust(wspace=0.5)
    fig.suptitle('Scatter plots of Predicted vs. Actual Values with Diagonal Line')

    st.pyplot(fig)


    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from math import sqrt

    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_rmse = sqrt(lr_mse)
    lr_mae = mean_absolute_error(y_test, lr_predictions)

    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_predictions)

    dt_mse = mean_squared_error(y_test, dt_predictions)
    dt_rmse = sqrt(dt_mse)
    dt_mae = mean_absolute_error(y_test, dt_predictions)

#Comparison graph of errors of the 3 models.

    data = {
      'Model': ['Linear Regression', 'Random Forest', 'Decision Tree'],
      'MSE': [lr_mse, rf_mse, dt_mse],
      'RMSE': [lr_rmse, rf_rmse, dt_rmse],
      'MAE': [lr_mae, rf_mae, dt_mae]}

    df_model_errors = pd.DataFrame(data)

# Plot Mean Squared Error
    fig_mse = px.scatter(df_model_errors, x='Model', y='MSE', color='Model', title='Mean Squared Error for all three models',
                     labels={'Model': 'Model', 'MSE': 'Mean Squared Error'},
                     color_discrete_map={'Linear Regression': 'red', 'Random Forest': 'blue', 'Decision Tree': 'green'})
    st.plotly_chart(fig_mse)

# Plot Root Mean Squared Error
    fig_rmse = px.scatter(df_model_errors, x='Model', y='RMSE', color='Model', title='Root Mean Squared Error for all three models',
                      labels={'Model': 'Model', 'RMSE': 'Root Mean Squared Error'},
                      color_discrete_map={'Linear Regression': 'red', 'Random Forest': 'blue', 'Decision Tree': 'green'})
    st.plotly_chart(fig_rmse)

# Plot Mean Absolute Error
    fig_mae = px.scatter(df_model_errors, x='Model', y='MAE', color='Model', title='Mean Absolute Error for all three models',
                     labels={'Model': 'Model', 'MAE': 'Mean Absolute Error'},
                     color_discrete_map={'Linear Regression': 'red', 'Random Forest': 'blue', 'Decision Tree': 'green'})
    st.plotly_chart(fig_mae)


