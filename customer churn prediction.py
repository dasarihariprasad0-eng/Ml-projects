import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = {"tenure_months":[18,12,3,5,6,7,8,8,9,10],"monthlycharges":[87,80,10,19,21,25,29,29,32,34],"fiber_optic":[0,0,0,1,0,1,1,0,0,1],"churn":[0,0,1,0,1,1,1,0,1,0] }
d = pd.DataFrame(data)
x = d.drop("churn",axis = 1)
y = d["churn"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train,y_train)
try:
    tenure = float(input("enter no.of months:"))
    monthcharge = float(input("enter monthly charge:"))
    opticfiber = int(input("has optic fiber ? if yes :1 if no:0:"))
    user_data = [[tenure,monthcharge,opticfiber]]
    pre = model.predict(user_data)
    if pre[0] == 1:
        print("high risk")
    else:
        print("low risk")
except ValueError:
    print("enter a valid numbers")
    


    