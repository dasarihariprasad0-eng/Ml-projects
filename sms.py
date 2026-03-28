from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
categories = ['sci.space', 'alt.atheism']
data = fetch_20newsgroups(subset = "all",categories = categories)
text = data.data
label = data.target
x_train,x_test,y_train,y_test = train_test_split(text,label,test_size = 0.4,random_state = 32)
model = Pipeline([("tfidf",TfidfVectorizer()),("nb",MultinomialNB())])
model.fit(x_train,y_train)
pre = model.predict(x_test)
print(f"accuracy: {accuracy_score(y_test,pre):.2f}")
print(classification_report(y_test,pre))


