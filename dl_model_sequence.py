import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_csv('Bangalore.csv')

# One hot encoding of Location feature
data_encoded = pd.get_dummies(data['Location'], prefix='Location')


# Join the encoded DataFrame with the original DataFrame
data = data.join(data_encoded)

# Drop original string column Location
data = data.drop('Location', axis=1)


# Split data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Define the model architecture
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model on the test set
predict = model.predict(X_test)

print(predict.shape)
print(y_test.shape)
# Error 

Error = mean_absolute_error(y_test,predict)
print(Error)


