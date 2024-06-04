import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Example training data (replace with your actual training data)
data = {
    'year': [2015, 2016, 2017, 2018, 2019],
    'showroom_price': [500000, 550000, 600000, 650000, 700000],
    'kilometer_driven': [40000, 30000, 20000, 10000, 5000],
    'owner_count': [2, 1, 1, 2, 1],
    'fuel_type_Diesel': [0, 1, 0, 1, 0],
    'fuel_type_Petrol': [1, 0, 1, 0, 1],
    'seller_type_Dealer': [1, 1, 0, 0, 1],
    'seller_type_Individual': [0, 0, 1, 1, 0],
    'transmission_Manual': [1, 1, 0, 0, 1],
    'transmission_Automatic': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(df)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Scaler saved as 'scaler.pkl'.")
