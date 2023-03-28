import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


#  Import data from csv
data = pd.read_csv('Bangalore.csv')

# print(data.dtypes)
# Location is not float or int

# One hot encoding - remove Location columns and replace it with multiple columns with 0 and 1, one column for each location 
data_encoded = pd.get_dummies(data['Location'],prefix='Location')

data = data.join(data_encoded)

data = data.drop('Location', axis=1)

# Split data into feature and target 
X = data.drop('price', axis=1)
y = data['price']



# Split the data into training and test 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)



# Import and train model - Random forest regressor 
model = RandomForestRegressor()


# Model training or Model fitting 

model.fit(X_train,y_train)


# predict 

predict = model.predict(X_test)

# Error 

Error = mean_absolute_error(y_test,predict)
print(Error)