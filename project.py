import streamlit as st 
import matplotlib.pyplot as plt 
import time
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras

st.title("Ad Click Prediction")
st.set_option('deprecation.showPyplotGlobalUse', False)
#add a sidebar
st.sidebar.subheader("Visualization setup")
#classifier_name=st.sidebar.selectbox("select",["Logistic Regression","Naive Bayes","DecisionTree"])
int_val1 = st.sidebar.number_input('Daily Time Spent on Site', min_value=1, max_value=100, value=5, step=1)
int_val2 = st.sidebar.number_input('Age', min_value=1, max_value=100, value=5, step=1)
int_val3 = st.sidebar.number_input('Area Income', min_value=1, max_value=100000, value=5, step=1)
int_val4 = st.sidebar.number_input('Daily Internet Usage', min_value=1, max_value=500, value=5, step=1)
int_val5 = st.sidebar.number_input('Gender', min_value=0, max_value=1)
int_val6 = st.sidebar.number_input('City Codes', min_value=1, max_value=10000, value=5, step=1)

int_val7 = st.sidebar.number_input('Country Codes', min_value=1, max_value=100000, value=5, step=1)

int_val8 = st.sidebar.number_input('Month', min_value=1, max_value=12, value=5, step=1)
int_val9 = st.sidebar.number_input('Hour', min_value=1, max_value=12, value=5, step=1)

uploaded_file=st.sidebar.file_uploader( 
    label="upload your Csv or Excel file",
    type=["csv","xlsx"])


if  st.sidebar.button("Process"):
    if uploaded_file is not None:
        print(uploaded_file)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        df = pd.read_excel(uploaded_file)
try:
    st.write(df)
except Exception as e:
    print(e)
    st.warning("Please upload an image before proceeding!")

numeric_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage' ]
categorical_columns = [ 'Ad Topic Line', 'City', 'Male', 'Country', 'Clicked on Ad' ]



try:
    #st.write("age group does the dataset majorly consist of?")
    plt.figure(figsize=(10,7))  
    sns.distplot(df['Age'], bins = 20, kde=True, hist_kws=dict(edgecolor="k", linewidth=1))
except NameError:
    pass
st.pyplot()

st.write("Here, we can see that most of the internet users are having age in the range of 26 to 42 years.")
st.write(f"Age of the oldest person: {df['Age'].max()},Years")
st.write(f"Age of the youngest person:' {df['Age'].min()}, Years")
st.write(f"Average age in dataset:' {df['Age'].mean()}, 'Years")

st.title("What is the income distribution in different age groups?")
sns.jointplot(x='Age', y='Area Income', color= "green", data= df)
st.pyplot()

st.title("Which age group is spending maximum time on the internet?")
sns.jointplot(x='Age', y='Daily Time Spent on Site', data= df)
st.pyplot()
st.write("Which gender has clicked more on online ads?")

st.write(df.groupby(['Male','Clicked on Ad'])['Clicked on Ad'].count().unstack())
st.title("Maximum number of internet users belong to which country in the given dataset?")
st.write(pd.crosstab(index=df['Country'],columns='count').sort_values(['count'], ascending=False))

st.title("What is the relationship between different features?")
sns.pairplot(df, hue='Clicked on Ad')
st.pyplot()
st.title("Data Cleaning")
sns.heatmap(df.isnull(), yticklabels=False)
st.pyplot()
st.write("""
As we see, we don't have any missing data
Considering the 'Advertisement Topic Line', we decided to drop it. In any case, if we need to extract any form of interesting data from it, we can use Natural Language Processing.
As to 'City' and the 'Nation', we can supplant them by dummy variables with numerical features, Nonetheless, along these lines we got such a large number of new highlights.
Another methodology would be thinking about them as a categorical features and coding them in one numeric element.
Changing 'Timestamp' into numerical value is more complicated. So, we can change ‘Timestamp’ to numbers or convert them to spaces of time/day and consider it to be categorical and afterwards we converted it into numerical values. And we selected the month and the hour from the timestamp as features
""")
df['City Codes']= df['City'].astype('category').cat.codes
df['Country Codes'] = df['Country'].astype('category').cat.codes
df['Month'] = df['Timestamp'].apply(lambda x: x.split('-')[1])
df['Hour'] = df['Timestamp'].apply(lambda x: x.split(':')[0].split(' ')[1])


X = df.drop(labels=['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis=1)
Y = df['Clicked on Ad']
model = keras.Sequential([keras.layers.Flatten(input_shape=(9,)),
keras.layers.Dense(16, activation=tf.nn.relu),
keras.layers.Dense(16, activation=tf.nn.relu),
keras.layers.Dense(1, activation=tf.nn.sigmoid)])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X = np.asarray(X).astype(np.float32)
Y = np.asarray(Y).astype(np.float32)
model.fit(X, Y, epochs=200, validation_split = .2)
result=model.predict([[int_val1,int_val2,int_val3,int_val4,int_val5,int_val6,int_val7,int_val8,int_val9]])
st.title("Result of prediction")
st.write(result)


# def get_classifier(clf_name):
#     if clf_name == "Logistic Regression":
#         clf = LogisticRegression()
#     elif clf_name == "Naive Bayes":
#         clf = GaussianNB()
#     else:
#         clf_name = "DecisionTree"
#         clf = DecisionTreeClassifier()
#     return clf 

# clf = get_classifier(classifier_name)
# X = df.drop(labels=['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis=1)
# Y = df['Clicked on Ad']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)
# clf.fit(X_train,Y_train)
# y_pred = clf.predict(X_test)
# acc = accuracy_score(Y_test,y_pred)
# st.write(f"classifier = {classifier_name}")
# st.write(f"accuracy = {acc}")  


# st.write("The prediction")

# st.write(f"result : {clf.predict([[int_val1,int_val2,int_val3,int_val4,int_val5,int_val6,int_val7,int_val8,int_val9]])}")

