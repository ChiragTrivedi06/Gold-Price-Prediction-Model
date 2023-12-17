import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('gold_price_model.sav', 'rb'))


def Gold_Price_Prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction


#main function
def main():
    
    
    # giving title
    st.title('Gold Price Preditiion Web App')
    
    
    # Taking input from user
    SPX = st.text_input('SPX Value:')
    USO = st.text_input('USO Value :')
    SLV = st.text_input('SLV Value :')
    EUR_USD = st.text_input('EUR/USD value :')
    
    
    Final_Price = ''
    
    # creating Button
    if st.button('Gold Price'):
        Final_Price = Gold_Price_Prediction([SPX,USO,SLV,EUR_USD])
        
    st.success(Final_Price)
    
    
    
if __name__ == '__main__':
    main()    
        
    
    
    
    
    
    