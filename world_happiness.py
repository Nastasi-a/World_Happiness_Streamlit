import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

df=pd.read_csv("merged_happiness_dataframe.csv")
df["year"] = df["year"].astype(int)
st.title("World Happiness Report")
st.sidebar.title("Table of contents")
pages=["Framework", "Visualization", "Modelling", "Comparison", "Interpretation", "Outlook"]
page=st.sidebar.radio("Go to", pages)

st.sidebar.write("\n\n\n\n\n")
st.sidebar.subheader("Team 🚀") 
st.sidebar.write("- [**Anastasiia Burtseva**](https://www.linkedin.com/in/anastasiia-burtseva-69bba9289/)")
st.sidebar.write("- [**Annika Heintz-Saad**](https://www.linkedin.com/in/annika-heintz-saad-79791b72/)")
st.sidebar.write("- [**Belal Mahmud**](https://www.linkedin.com/in/belal-mahmud-394b22113/)")


#Creation of Framework page

if page == pages[0] :
  st.image('images/happiness.png')
  st.header("Framework 📐")

  st.write("We started with two dataframes in our project:")

  st.write("- **world-happiness-report-2021:** This dataset includes data on world happiness for the year 2021.")
  st.write("- **world-happiness-report:** This dataset includes information about world happiness before 2021.")

  st.write("The goal of the project was to analyze the world's happiness score and how it is influenced by certain indicators. We observed how it has developed in the past and predicted the future trend. The data is freely available for analysis [here](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021).")

  st.write("In the first step we have explored the data, cleaned and merged our dataframes. In this streamlit app, we are working with the merged and cleaned dataset called **merged_happiness_dataframe**.")

  st.subheader('Shape')
  st.write(df.shape)
  st.write("Our dataset consists of 2083 rows and 12 columns")
  
  st.subheader('Columns')
  
  table_data = {
    "Data": ["🌍", "🌐", "📅", "⭐", "💰", "🤝", "🏥", "🔓", "🎁", "🔍", "😊", "😞"],
    "Column": ["Country name", "Regional indicator", "Year", "Ladder score", "Logged GDP per capita",
               "Social support", "Healthy life expectancy", "Freedom to make life choices", "Generosity",
               "Perceptions of corruption", "Positive affect", "Negative affect"],
    "Description": ["Name of the country.",
                    "Region where the country is located.",
                    "Year of the data.",
                    "Happiness score based on the Gallup World Poll (GWP). It represents respondents' life evaluations on a scale from 0 to 10.",
                    "GDP per capita adjusted for population size.",
                    "Measure of having someone to count on in times of trouble.",
                    "Average number of healthy years a person can expect to live.",
                    "Measure of satisfaction with personal freedom.",
                    "Measure of charitable giving relative to GDP per capita.",
                    "Measure of perceived corruption in government and businesses.",
                    "Average of happiness, laughter, and enjoyment experienced.",
                    "Average of worry, sadness, and anger experienced."]}
  
  
  df_columns = pd.DataFrame(table_data)
  st.table(df_columns.set_index("Data"))
  st.subheader('Dataframe')

  st.dataframe(df.head(10))

  if st.checkbox("Show NA") :
    st.dataframe(df.isna().sum())


#Creation of Data Visualization page

if page == pages[1] : 
  st.header("Data Visualization 🎨")

  st.subheader('Statistics')
  st.dataframe(df.describe())

  st.write("This table presents statistical summaries for variables of our dataframe the years 2005 to 2021. Here we provide an overview of the distribution and central tendencies of the variables related to happiness scores across different countries and years.")
  st.write("\n\n\n\n\n")

#Bar plot to show the distribution of the Ladder score in 2021.

  df_2021 = df[df['year'] == 2021]
  df_filtered_1 = df_2021.sort_values(by='Ladder score', ascending=False) #Creating filtered DataFrame with sorted values according to our needs

  fig = px.bar(df_filtered_1, x='Country name', y='Ladder score')
  fig.update_layout(title='Ladder Score in 2021 by Country')
  st.plotly_chart(fig)
  st.write("This barplot shows the Ladder Score in 2021 for every country available in the dataset.")

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

  st.write("These are the Top10 countries with the highest Ladder Score and the Last10 countries with the lowest Ladder Score.")

#Creating 'Ladder score category' variable in the 2021 dataframe with the values 'ladder score low', 'ladder score medium' and 'ladder score high' with the help of the quantiles.

  st.write("\n\n\n\n\n")
  st.write('**World map with Ladder Score Categories**')

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
  st.write("You can see some white parts on the world map. Those are the countries that are missing completely in the data set as we had to delete them due to too many missing values. Countries filled with green have the highest Ladder scores, yellow color displays middle values, and red - lowest. \n\n You can get a sense from these graphs that most countries with a high Ladder score are developed countries / high-income countries, whereas most countries with low Ladder score are developing or least developed countries. Therefore, we can assume that GDP per capita plays a big role for the Ladder score.")
  
  
  st.write("\n\n\n\n\n")
  st.write('**The correlation heatmap**')

#Creating a correlation matrix
  cor = df.iloc[:, 1:12].corr()
#Creating a heatmap
  fig, ax = plt.subplots(figsize =(8,8))
  sns.heatmap(cor, annot = True, ax = ax, cmap = 'Spectral')
  plt.title('The heatmap for the world happiness report')
  st.pyplot(fig)
  st.write("On the heatmap above we can observe the highest correlation between Healthy life expectancy and Logged GDP per capita. It can be explained, because a strong economy of a country leads people to live better and longer lives. \n We can also notice the most correlated variables to the Ladder score: Healthy life expectancy, Social Support and Logged GDP per capita. Hence, economics, social support and life expectancy influences on the Ladder score more than the other indicators. We can also observe the highest negative correlation among all the given variables between Perceptions of corruption and Freedom to make life choices, which means that corruption perceptions are inversely linked to the sense of freedom individuals and experience in making life decisions.")
  st.write("\n\n\n\n\n")

#Creating a filtered dataframe
  df_filtered_first = df[df['year'] == 2021]
  df_filtered = df_filtered_first.sort_values(by='Positive affect', ascending=True).tail(10)
  df_filtered.head(10)
#Creating a horizontal barplot
  df_filtered = df_filtered_first.sort_values(by='Positive affect', ascending=False).head(10) #Creating filtered DataFrame with sorted values according to our needs
  fig = px.bar(df_filtered,
             x='Positive affect',
             y='Country name',
             orientation='h',
             title='Countries with the highest Positive affect in 2021',
             labels={'Positive affect': 'Positive Affect', 'Country name': 'Country Name'},
             height=500,
             width=800,
             color='Positive affect',
             color_continuous_scale='greens',
             opacity=0.8
            )
  fig.update_layout(yaxis=dict(autorange="reversed")) # Reverse the order of countries on the y-axis
  st.plotly_chart(fig)
  st.write("The graph illustrates the top 10 countries with the highest positive affect scores in 2021, showcasing their relative levels of happiness and well-being.")

  st.write("\n\n\n\n\n")

  df_filtered = df_filtered_first.sort_values(by='Negative affect', ascending=False).head(10)#Creating filtered DataFrame with sorted values according to our needs
#Creating a horizontal barplot
  fig = px.bar(df_filtered,
             x='Negative affect',
             y='Country name',
             orientation='h',
             title='Countries with the highest Negative affect in 2021',
             labels={'Negative affect': 'Negative Affect', 'Country name': 'Country Name'},
             height=500,
             width=800,
             color='Negative affect',
             color_continuous_scale='oranges',
             opacity=0.8
            )

  fig.update_layout(yaxis=dict(autorange="reversed"))  # Reverse the order of countries on the y-axis
  st.plotly_chart(fig)
  st.write("The graph illustrates the top 10 countries with the highest negative affect scores in 2021, providing insights into the levels of distress or dissatisfaction experienced by their populations.")
  st.write("\n\n\n\n\n")
  st.write("**The pairplot which shows the correlation between Positive and Negative affect**")
  correlation_2021 = df_filtered_first[['Positive affect', 'Negative affect']]
  pairplot = sns.pairplot(correlation_2021, kind='reg')
  st.pyplot(pairplot.fig)
  st.write('On the pairplot above we can see a negative correlation between the Positive affect and the Negative affect. Thats why we can conclude that as Positive affect scores increase for a particular country, Negative affect scores tend to decrease, and vice versa.')
  
  st.write("\n\n\n\n\n")

  st.write("**Distribution of Ladder Score in 2021**")
#Plot to show the distribution of the ladder score in 2021.
  fig = px.box (df_2021, x = 'Ladder score') 
  st.plotly_chart(fig)

  st.write("The graph displays the distribution of ladder scores in 2021, revealing an extreme value for one country's happiness level compared to others.")


  st.write("\n\n\n\n\n")
  
  st.write("**Development of the Ladder Score over years for each Regional Indicator**")
  
  fig, ax = plt.subplots(figsize=(12, 6))
  sns.lineplot(x='year', y='Ladder score', hue='Regional indicator', errorbar=None, data=df, marker='o', ax=ax)
  ax.set_xlabel('Year')
  ax.set_ylabel('Ladder Score')
  ax.legend(title='Regional Indicator', loc='center left', bbox_to_anchor=(1, 0.5))
  st.pyplot(fig) 

  st.write("Various regions began recording happiness scores at different times. Sub-Saharan, Commonwealth of Independent States, and South Asian countries started in 2006. North America leads in happiness scores, followed closely by Latin America and the Caribbean. In contrast, Sub-Saharan and South Asian countries have the lowest rankings, with Sub-Saharan nations leading in this aspect.")

  st.write("\n\n\n\n\n")
  st.write("**Distribution of the Ladder Score by Regional Indicator**")
  regional_indicators = df['Regional indicator'].unique()
  fig, ax = plt.subplots(figsize=(12, 6))
  sns.boxplot(x='Regional indicator', y='Ladder score', data=df, hue='Regional indicator', width=0.8, order=regional_indicators, ax=ax)
  ax.set_xlabel('Regional Indicator')
  ax.set_ylabel('Ladder Score')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  ax.get_legend().remove()
  st.pyplot(fig)

  st.write("The graph above displays the happiness ladder score based on the regional indicator. The results are consistent with previous findings, showing minimal deviation. However, some extreme values are noticeable in the graph due to variations in the starting year of recording the ladder score across different regions in our dataset.")


#Creation of Modelling page

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


if page == pages[2] : 
  st.header("Modelling ⚙️")

# Display R² score on the train and test set

  choice = ['Linear Regression', 'Random Forest', 'Decision Tree']
  option = st.selectbox('Choice of the model', choice)
  st.write('**The chosen model is :**', option)

  if option == 'Linear Regression':
    st.write("\n\n\n")
    st.write('R² Score on the train set with Linear Regression:', lr.score(X_train, y_train))
    st.write('R² Score on the test set with Linear Regression:', lr.score(X_test, y_test))

    st.write("\n\n\n")

    lr_coefficients = lr.coef_[0] 
    fig = px.bar(lr_coefficients)
    fig.update_layout(title='Coefficients')
    st.plotly_chart(fig)

    st.write("The coefficients show how the target variable (y) changes with a one-unit increase in the explanatory variable (x), while keeping all other variables constant. For instance, a coefficient of -0.00238972 means y decreases by this amount with a one-unit increase in x. Conversely, a coefficient of 0.85110927 signifies y increases by this amount with the same change in x. These coefficients represent the relationships between the target and explanatory variables.")
    
  elif option == 'Random Forest': 
    st.write("\n\n\n")
    st.write('R² Score on the train set with Random Forest:', rf.score(X_train, y_train))
    st.write('R² Score on the test set with Random Forest:', rf.score(X_test, y_test))
    st.write("\n\n")
    st.write("The Random Forest model seems to be overfitted. It performs really well on the training set but poorly on the unseen data.")
   
    st.write("\n\n\n")
   
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    rf_feature_importance_df = pd.DataFrame({
      'Feature': X_train.columns,
      'Importance': rf.feature_importances_})

    rf_feature_importance_df = rf_feature_importance_df.sort_values(by='Importance', ascending=False)

    fig = px.bar(rf_feature_importance_df, x='Feature', y='Importance', labels={'Importance': 'Importance', 'Feature': 'Feature'}, title='Random Forest Feature Importances')
    st.plotly_chart(fig)

    importance_threshold = 0.0019

    selected_features = rf_feature_importance_df[rf_feature_importance_df['Importance'] >= importance_threshold]['Feature']

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    rf_selected = joblib.load('trained_rf_selected.joblib')

    rf_selected_predictions = rf_selected.predict(X_test_selected)

    st.write('R² Score on the train set with Random Forest with reduced features:', rf_selected.score(X_train_selected, y_train))
    st.write('R² Score on the test set with Random Forest with reduced features:', rf_selected.score(X_test_selected, y_test))

    st.write("\n\n\n")
    
    st.write('**Shape before and after feature reduction:**')
    st.write(X_train.shape)
    st.write(X_train_selected.shape)
    st.write("After the feature selection process only 15 features are left instead of 177.")

    rf_train_score = rf.score(X_train, y_train)
    rf_test_score = rf.score(X_test, y_test)

    rf_train_selected_score = rf_selected.score(X_train_selected, y_train)
    rf_test_selected_score = rf_selected.score(X_test_selected, y_test)

    labels = ['Random Forest Train', 'Random Forest Train w/ reduced features', 'Random Forest Test', 'Random Forest Test w/ reduced features']
    scores = [rf_train_score, rf_train_selected_score, rf_test_score, rf_test_selected_score]
    colors = ['blue', 'orange', 'blue', 'orange']

    fig = go.Figure(data=[go.Bar(x=labels, y=scores, marker_color=colors)])
    fig.update_layout(title='Performance on Train and Test Sets of Random Forest with all and reduced features',
                  xaxis_title='Dataset',
                  yaxis_title='R² Score',
                  yaxis=dict(range=[0, 1]))

    st.plotly_chart(fig)


    st.write("Reducing the features of Random Forest leads to almost the same results in the R² score. And it lowers the complexity of the model immensly.")

    from sklearn.metrics import mean_absolute_error

    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_selected_mae = mean_absolute_error(y_test, rf_selected_predictions )

    labels = ['rf_mae', 'rf_selected_mae']
    mae_values = [rf_mae, rf_selected_mae]
    colors = ['blue', 'orange']

    fig = go.Figure(data=go.Scatter(x=labels, y=mae_values, mode='markers', marker=dict(color=colors)))
    fig.update_layout(title='MAE of Random Forest before and after feature reduction',
                  xaxis_title='Model',
                  yaxis_title='Mean Absolute Error',
                  showlegend=False,
                  template='plotly_white')

    st.plotly_chart(fig)

    st.write ("The MAE is now a little bit higher after getting rid of a number of features.")

    X_train_selected = pd.DataFrame(X_train_selected)
    X_test_selected = pd.DataFrame(X_test_selected)

    rf_feature_importance_df_selected = pd.DataFrame({
        'Feature': X_train_selected.columns,
        'Importance': rf_selected.feature_importances_})

    rf_feature_importance_df_selected = rf_feature_importance_df_selected.sort_values(by='Importance', ascending=False)

    fig = go.Figure(go.Bar(x=rf_feature_importance_df_selected['Feature'],
                          y=rf_feature_importance_df_selected['Importance'],
                          marker_color='blue'))
    fig.update_layout(title='Random Forest Feature Importances after Feature Reduction',
                      xaxis_title='Feature',
                      yaxis_title='Importance',
                      template='plotly_white')
    st.plotly_chart(fig)

    st.write("The feature importance graph after feature reduction looks very similar to the original one. That means that the feature reduction has effectively selected the most important features from the dataset. The model seems to be performing well after feature reduction therefore we will go with the new model with only 15 features.")

  else: 
    st.write("\n\n\n")
    st.write('R² Score on the train set with Decision Tree:', dt.score(X_train, y_train))
    st.write('R² Score on the test set with Decision Tree:', dt.score(X_test, y_test))

    from sklearn.tree import plot_tree
    
    X_train = pd.DataFrame(X_train)
    feature_names = list(X_train.columns)
    plt.figure(figsize=(20,10))
    plot_tree(dt,filled=True, feature_names=feature_names)
    st.pyplot(plt.gcf())   
    

#Creation of Comparison page

if page == pages[3] : 
  st.header("Comparison of models ⚖️")

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

  st.write ("Random Forest performed best on the test set, however it seems to be overfitted to the training set. Linear Regression also performed well in comparison to Decision Tree.")


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

  st.write ("The 45-degree diagonal line allows us to assess the accuracy of our models' predictions. It's evident that the dots corresponding to the Random Forest model are the closest to this line, indicating a higher level of accuracy.")

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

  st.write("The accuracy of our predictions correlates with lower error values. Random Forest and Linear Regression have lower errors than Decision Tree, therefore we can say those two are more accurate.")
    
  st.write("\n\n\n")
  st.subheader ("Hyperparameter tuning")
  st.write("We have carried out **hyperparameter tuning**. The grid we used didn’t give us expected result. Therefore, we have decided to proceed with the original models.")

#Creating graphs to show errors after hyperparameter tuning
  models = ['Linear Regression', 'Random Forest','Decision Tree']
  lr_mse = 0.16
  rf_mse = 0.14
  dt_mse = 0.33
  lr_mae = 0.29
  rf_mae = 0.28
  dt_mae = 0.45
  lr_rmse = 0.39
  rf_rmse = 0.37
  dt_rmse = 0.57
  mae_best_lr = 0.30
  mae_best_rf = 0.34
  mae_best_dt = 0.57
  mse_best_lr = 0.16
  mse_best_rf = 0.2
  mse_best_dt = 0.5
  rmse_best_lr = 0.4
  rmse_best_rf = 0.4
  rmse_best_dt = 0.4

  mse_before = [lr_mse, rf_mse, dt_mse]
  rmse_before = [lr_rmse, rf_rmse, dt_rmse]
  mae_before = [lr_mae, rf_mae, dt_mae]

  mae_after = [mae_best_lr, mae_best_rf, mae_best_dt]
  mse_after = [mse_best_lr, mse_best_rf, mse_best_dt]
  rmse_after = [rmse_best_lr, rmse_best_rf, rmse_best_dt]

  fig_mae = go.Figure()
  fig_mae.add_trace(go.Scatter(x=models, y=mae_before, mode='markers', marker=dict(color='blue'), name='MAE Before Tuning'))
  fig_mae.add_trace(go.Scatter(x=models, y=mae_after, mode='markers', marker=dict(color='red'), name='MAE After Tuning'))
  fig_mae.update_layout(title='MAE Before and After Hyperparameter Tuning', xaxis_title='Model', yaxis_title='MAE')

  fig_mse = go.Figure()
  fig_mse.add_trace(go.Scatter(x=models, y=mse_before, mode='markers', marker=dict(color='blue'), name='MSE Before Tuning'))
  fig_mse.add_trace(go.Scatter(x=models, y=mse_after, mode='markers', marker=dict(color='red'), name='MSE After Tuning'))
  fig_mse.update_layout(title='MSE Before and After Hyperparameter Tuning', xaxis_title='Model', yaxis_title='MSE')

  fig_rmse = go.Figure()
  fig_rmse.add_trace(go.Scatter(x=models, y=rmse_before, mode='markers', marker=dict(color='blue'), name='RMSE Before Tuning'))
  fig_rmse.add_trace(go.Scatter(x=models, y=rmse_after, mode='markers', marker=dict(color='red'), name='RMSE After Tuning'))
  fig_rmse.update_layout(title='RMSE Before and After Hyperparameter Tuning', xaxis_title='Model', yaxis_title='RMSE')

  st.plotly_chart(fig_mae, use_container_width=True)
  st.plotly_chart(fig_mse, use_container_width=True)
  st.plotly_chart(fig_rmse, use_container_width=True)


#Creation of Interpretation page

if page == pages[4] : 
  st.header("Interpretation of results 🔍")

  st.write("In summary, we explored three distinct supervised learning models for our dataset: **Linear Regression, Random Forest Regression, and Decision Tree Regression**. Upon assessing their performance, we opted to exclude the Decision Tree Regressor model due to its underwhelming R² score and error metrics")

  st.write("Both **Linear Regression and Random Forest** show good R² scores and minimal errors.")

  st.write("By reducing the **number of features** to 15, we managed to mitigate the complexity of the Random Forest model while maintaining comparable performance levels. Consequently, Random Forest emerges as the most suitable model for our objectives.")
  
  st.write("\n\n\n")
  st.image('images/Random-forest-prediction-scheme.png')
  st.write("[Picture Copyright](https://www.researchgate.net/figure/Random-forest-prediction-scheme_fig2_357828695)")


#Creation of Difficulties page

if page == pages[5] : 
  st.header("Recap and Outlook 📝")
 
  st.subheader ("Difficulties during the project")
  st.write("- **Absence of regional data for certain countries**: prompting us to seek data from alternative sources.")
  st.write("- **Lacked access to data for all countries for the entire time period (since 2005)**: We created data visualizations specifically for 2021, which is the most recent and data-enriched year.")
  st.write("- **Absence of data for certain countries**: we had to remove several rows containing numerous NaN values to avoid erroneous conclusions.")
  st.write("- **Issues during the encoding of our dataframe**: When encoding categorical features and creating a new dataframe, we encountered issues with inconsistent indices starting from 0. To resolve this, we used the reset_index function to ensure consistency when concatenating with the original dataframe containing numerical features.")

  st.write("\n\n\n")
  st.subheader ("Prediction of other variables")
  st.write("The dataset could be utilized to predict other variables such as:")
  st.write("- **Logged GDP per capita**")
  st.write("- **Social support**")
  st.write("- **Healthy life expectancy**")
  st.write("- **Freedom to make life choices**")
  st.write("- **Generosity**")
  st.write("- **Perceptions of corruption**")
  st.write("- **Positive affect Negative affect**")
  st.write("- **Negative affect**") 
  st.write("Each of these variables could be considered as the target variable in separate regression analyses.")
  st.write("\n\n\n")

  st.subheader ("Temporal analysis")
  st.write("It is feasible to go into a temporal analysis. We could observe trends and changes over time and provide valuable insights into the evolution of happiness scores and related indicators. Before doing it, we need to research and enrich the data for the given period of time.")
  st.image('images/graph.png')
  st.write("\n\n\n")

  st.subheader ("Feature engineering techniques")
  st.write("In continuation we can experiment with more diverse feature engineering techniques and exploring different feature selection methods  improve model performance and interpretability.")
  st.write("\n\n\n")

  st.subheader ("External data")
  st.write("It is also possible to incorporate external data such as environmental factors, more socio-economic indicators, lifestyle indicators such as work-life balance to see how they interact with existing variables in our analysis. It can provide more complex insights about the exploration of well-being.")
  st.image('images/4.jpeg', width = 150)

