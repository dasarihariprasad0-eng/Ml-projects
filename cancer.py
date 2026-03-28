from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 32)

scale = StandardScaler()
x_train_scale = scale.fit_transform(x_train)

x_test_scale = scale.transform(x_test)

model = SVC(kernel = "linear")
model.fit(x_train_scale,y_train)

y_pred = model.predict(x_test_scale)

accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy:{accuracy:.2f}")
cla = classification_report(y_test,y_pred)
print("classification:",cla)

