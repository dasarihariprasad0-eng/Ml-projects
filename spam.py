import pandas as pd
df = pd.read_csv("/storage/emulated/0/Download/spam.csv")
print(df)
print(df.groupby("Category").describe())
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
x = cv.fit_transform(df["Message"])
y = df["Category"]
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x,y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
print(model.score(x_test,y_test))
n = ["congratulations you have won 1300"]
new = cv.transform(n)
print(model.predict(new))



