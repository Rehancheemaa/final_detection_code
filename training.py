import pandas as pd
import joblib
import numpy as np

df = pd.read_csv("/Suicide_Detection.csv")
df2 = pd.read_csv("/Cyber_bullying_processed.csv")

df2.head()

df2 = df2.rename(columns={'text': 'text', 'label': 'class'})

unique_values = df2.iloc[:, 1].unique()

unique_values

labels_map={
    0:'non-bullt',
    1:"bully",
    3:'suicide',
    4:'non-suicide',
}

df.head()

df_old = df

df = df_old

df = df.drop(["Unnamed: 0"], axis = 1)

df

df.iloc[:, 1] = df.iloc[:, 1].map({'suicide': 3, 'non-suicide': 4})

df

df2

new_df = pd.concat([df, df2], ignore_index=True)

new_df

new_df.dropna(inplace=True)

#Naive Multinomial Bayes with Sklearn

X = new_df['text']
Y = new_df['class']

from sklearn.model_selection import train_test_split

#Train 75% of the data, 25% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)

#Convert words into bag of words model with sklearns CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_features = 5000) #limit to 5000 unique words, but room to mess around is here



X_train_counts = count_vect.fit_transform(X_train)
#print(count_vect.vocabulary_) #here is our bag of words
joblib.dump(count_vect, 'count_vect.pkl')
X_test = count_vect.transform(X_test) # note, this is not fit to the model

print(count_vect.vocabulary_)

#Naive Bayes

from sklearn.naive_bayes import MultinomialNB

a = Y_train.unique()

a

Y_train = np.array(Y_train, dtype=int)  # Convert to integers

#fit the training dataset on the NB classifier
Naive = MultinomialNB() #create model
Naive.fit(X_train_counts, Y_train)

import joblib
joblib.dump(Naive, 'naive_bayes_model.pkl')

import scipy

scipy.__version__

#Test Accuracy
from sklearn.metrics import accuracy_score

#predict the labels on validation dataset
predictions_NB = Naive.predict(X_test)

Y_test = np.array(Y_test, dtype=int)


#Use accuracy score fn to get accuracy. Very high accuracy b/c of assumption of independence
print("Accuracy Score:", accuracy_score(predictions_NB, Y_test)*100)

#Onion Test

# link: https://entertainment.theonion.com/drake-fans-accuse-kenny-chesney-of-manipulating-billboa-1843484082
onion = ["i need helpjust help me im crying so hard"]

onion_vec = count_vect.transform(onion) #create bag of words

predict_onion = Naive.predict(onion_vec) #apply it to trained model

print(predict_onion) #1 = fake news

from sklearn.feature_extraction.text import CountVectorizer

count_vect = joblib.load('count_vect.pkl')
Naive_loaded = joblib.load('naive_bayes_model.pkl')



text = ["really suck"]
text_vec = count_vect.transform(text)
# You can now use Naive_loaded to make predictions
predictions = Naive_loaded.predict(text_vec)  # Example prediction
print(predictions)
