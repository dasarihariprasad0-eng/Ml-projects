from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
x = housing.data
y = housing.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 32)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"mse:{mse:.4f}")
print(f"r2:{r2:.4f}")
