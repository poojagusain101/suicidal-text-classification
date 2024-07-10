import streamlit as st
import pickle
import nltk
import re
from nltk.stem import PorterStemmer


from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
# stemming
ps = PorterStemmer()

# Load the trained model and vectorizer
with open('LogReg_model2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer2.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("Suicidal Text Classification App")

st.write("This is a suicidal ideation classifier to determine if a text is likely to include suicidal thoughts or not.")

user_input = st.text_area("Enter your text here:")

# preprocess the input data
def classify_text(inp):
    inp = inp.lower() #convert to lower case
    inp = re.sub(r'[^\w\s]+','',inp) #remove punctuations
    inp = [word for word in inp.split() if word not in (stop_words)] #tokenize the sentence
    inp = ' '.join([ps.stem(i) for i in inp]) #stemming
    inputToModel = vectorizer.transform([inp]).toarray() #transform to vector form
    predict = model.predict(inputToModel) #Model prediction
    return predict;   

# print('Output : ', predict[0]) 

# Function to predict class
# def classify_text(text):
#     text_transformed = vectorizer.transform([text])
#     prediction = model_file.predict(text_transformed)
#     return "Class 1" if prediction[0] == 1 else "Class 0"

# if st.button("Classify"):
#     if user_input:
#         prediction = classify_text(user_input)
#         st.write(f"Prediction: **{prediction}**")
#     else:
#         st.write("Please enter some text to classify.")

#  Predict class
if st.button("Classify"):
    if user_input:
        prediction = classify_text(user_input)
        if prediction == "suicide":
            st.error(f"Prediction: **{prediction}**")
        else:
            st.success(f"Prediction: **{prediction}**")
        # st.write(f"Prediction: **{'Class 1' if prediction == 1 else 'Class 0'}**")
    else:
        st.write("Please enter some text to classify.")
