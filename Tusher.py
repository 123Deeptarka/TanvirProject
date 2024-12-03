# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:36:38 2024

@author: deeptarka.roy
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Install seaborn if not already installed
import streamlit as st 

# Set the font family to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
# Set the default text color to black
plt.rcParams["text.color"] = "black"

#File Import from drive



df2=pd.read_excel("C:/Users\Deeptarka.Roy\Desktop\GUI for Thesis\Fiber Flexural Strength.xlsx")
df2.head(10)

def user_input_features():
    D=st.sidebar.number_input("W/C \nRatio",value=0.2)
    LD=st.sidebar.number_input("Amount of Coarse \nAggregate",value=1.5)   
    fc =st.sidebar.number_input("Amount of Fine \nAggregate\n",value=2.5)
    fyl =st.sidebar.number_input("Admixture provided",value=0)
    fyt =st.sidebar.number_input("% of Fiber",value=0.1)
    pl =st.sidebar.number_input("Fiber Type",value=1)
    pt =st.sidebar.number_input("Aspect Ratio (l/d)",value=50)
    Ny =st.sidebar.number_input("Fiber tensile strength",value=100)
    data={"W/C \nRatio":D,"Amount of Coarse \nAggregate":LD,"Amount of Fine \nAggregate\n":fc,"Admixture provide":fyl,"% of Fiber":fyt,"Fiber Type":pl,"Aspect Ratio (l/d)":pt,"Fiber tensile strength":Ny}
    features=pd.DataFrame(data,index=[0])
    features_round=features.round(2)
    return features_round

st.markdown('<h1 style="font-size: 40px; font-weight: bold;"> GUI for Flexural Strength Prediction</h1>', unsafe_allow_html=True)
st.header("Specified Input Parameters")
Data=user_input_features()
#prediction=model.predict(Data)

#df1 is for log normalization
x = df2.iloc[:, :-1].values
y = df2.iloc[:, -1].values
x = pd.DataFrame(x)
y = pd.DataFrame(y)
#print(x.head())
#print(y.head())from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
from sklearn.preprocessing import StandardScaler
sz = StandardScaler()
x_train = sz.fit_transform(x_train)
x_test = sz.transform(x_test)
#print(x_train)
#print(x_test)#Feature Count for Adjusted R2
features = list(df2.columns.values)
print(features)

#Calling Adjusted R2
def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)

#Calling other necessary Lib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#import pickle

#Evaluation DataFrame
evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'RMSE(train)':[],
                           'R-squared (train)':[],
                           'Adj R-squared (train)':[],
                           'MAE (train)':[],
                           'RMSE (test)':[],
                           'R-squared (test)':[],
                           'Adj R-squared (test)':[],
                           'MAE(test)':[],
                           '10-Fold Cross Validation':[]})

evaluation2 = pd.DataFrame({'Model': [],
                           'Test':[],
                           '1':[],
                           '2':[],
                           '3':[],
                           '4':[],
                           '5':[],
                           '6':[],
                           '7':[],
                           '8':[],
                           '9':[],
                           '10':[],
                           'Mean':[]})
import xgboost as xgb

#lr = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, subsample=0.7,
#                           colsample_bytree=0.5, max_depth=24, min_child_weight=2)
lr = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, subsample=0.5,
                           colsample_bytree=0.5, max_depth=4, min_child_weight=2)
y_train_1d = np.ravel(y_train)
y_test_1d = np.ravel(y_test)

lr.fit(x_train, y_train_1d)
y_pred_test = lr.predict(x_test)
y_pred_train = lr.predict(x_train)


y_pred_GUI=lr.predict(Data)


# Apply styling to dataframe
#st_df = Data.style.set_table_styles(style).format("{:.3f}").hide(axis="index")
# Convert to HTML without index and display
# Reset the index and drop it to remove the index column
Data_no_index = Data.reset_index(drop=True)


# Convert the DataFrame to HTML without the index
html = Data_no_index.to_html(index=False)

# Apply custom CSS to the HTML table
html = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 6px solid #484848 ; !important;  /* Thicker border with !important */
        padding: 3px;
        text-align: left;
    }}
    th {{
        font-size: 20px;
        font-weight: bold;
        color: #484848;
        border:2px solid #484848;
    }}
    td {{
        font-size: 16px;
        font-weight: bold;
        color: #484848;
    }}
</style>
{html}
"""

# Display the HTML table
st.markdown(html, unsafe_allow_html=True)


st.header("Flexural Strength Prediction Result:")

#st.subheader("Prediction Result:")
#st.write(y_pred_GUI)
prediction_value = np.round(y_pred_GUI[0], 1)
st.markdown(f'<h2 style="font-size:24px;">{y_pred_GUI} MPa</h2>', unsafe_allow_html=True)




