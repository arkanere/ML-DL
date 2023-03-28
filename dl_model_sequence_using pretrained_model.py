
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import load_model


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


# import pretrained model 
model = load_model('trained_model.h5')


# Evaluate the model on the test set
predict = model.predict(X_test)


# Error 
Error = mean_absolute_error(y_test,predict)
print(Error)


