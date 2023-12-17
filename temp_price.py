import pandas as pd
import numpy as np
import pickle 


# loading the saved model
loaded_model = pickle.load(open('gold_price_model.sav', 'rb'))

input_data = (	2352.949951,	10.680000,	17.320000,	1.067247)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
