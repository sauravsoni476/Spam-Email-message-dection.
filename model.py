import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib

msg = pd.read_csv("spam.csv", encoding='latin-1')

msg.head()

msg.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1, inplace=True)

msg.rename(columns={'v1': 'category', 'v2': 'message'}, inplace=True)

#msg.groupby('Class').describe()

#msg['spam'] = msg['category'].apply(lambda x: 1 if x == 'spam' else 0)

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(msg.message, msg.spam, test_size = 0.30)

#print(f' x train is=   {X_train.shape},\n x test size is ==  {X_test.shape},\n y train size is== {y_train.shape},\n y test size is == {y_test.shape}')
msg.rename(columns = {'class':'Class'}, inplace = True)
msg.head()
msg['label'] = msg['Class'].map({'ham': 0, 'spam': 1})
x = msg.message
y = msg.label

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

v = CountVectorizer()

X_train_count = v.fit_transform(x)
pickle.dump(v, open('tranform.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X_train_count, y, test_size=0.3, random_state=42)

#from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)
filename = 'nlp_model.pkl'
pickle.dump(model, open(filename, 'wb'))


## prediction this type of mail example own