import streamlit as st
import pickle as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
def get_clean_data():
    df= pd.read_csv("Academic_Performance_data.csv")
    return df

def add_sidebar():
    st.sidebar.header('Features')
    df = get_clean_data()
    
    slider_labels = [
        ('Hours Studied(Hs)','Hours Studied'),
        ('Previous Scores(Ts)','Previous Scores'),
        ('Sleep Hours(Sh)','Sleep Hours'),
        ('Sample Question(Sq)','Sample Question'), 
        ('Papers Practiced(PP)','Papers Practiced')]
        

 

    
    input_dict ={}
    for label, key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(df[key].max()),
            value=float(df[key].max())
        )
    return input_dict

def get_scaled_values(input_dict):
    df=get_clean_data()
    
    x=df.drop(['diagnosis'], axis=1)
    
    scaled_dict ={}
    
    for key, value in input_dict.items():
        max_val=x[key].max()
        min_val=x[key].min()
        scaled_value =(value-min_val)/(max_val -min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict
def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  features = ['Hours Studied','Previous Scores','Sleep Hours','Sample Question',
              'Papers Practiced']
              
	
             
              

  fig = go.Figure()
		

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['Hours Studied'], input_data['Previous Scores'],input_data['Sleep Hours'],
          input_data['Sample Question'],input_data['Papers Practiced']],
         
        
        theta=features,
        fill='toself',
        name='Mean Value'
  ))
 

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

def train_model():
    df = get_clean_data()
    X = df.drop(['Performance Index'], axis=1)
    y = df['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Train model when app starts
model = train_model()

def add_predictions(input_data):
    scaled_input = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(scaled_input)

    st.subheader("Academic Performance prediction")
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

    st.write("Probability of being benign:", model.predict_proba(scaled_input)[0][0])
    st.write("Probability of being malignant:", model.predict_proba(scaled_input)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def plot_metrics(metrics_list):
    fig, ax = plt.subplots()
    
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay(model, x_test, y_test)
        st.pyplot(fig1)


    if 'ROC Curve' in metrics_list:
        fig2, ax = plt.subplots()
        st.subheader("ROC Curve") 
        RocCurveDisplay(model, x_test, y_test)
        st.pyplot(fig2)
          
          
      
    if 'Precision-Recall Curve' in metrics_list:
        fig3, ax = plt.subplots()
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay(model, x_test, y_test)
        st.pyplot(fig3)
        
           
      
        
        
       

        


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female -doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data=add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    col1, col2 = st.columns([4,1])

    with col1:  
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)



if __name__ == '__main__':
    main()
