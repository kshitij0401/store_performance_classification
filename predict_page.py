import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
     

    return data

data = load_model()

dec_tree = data["model"]
le_car = data["le_car"]
le_location = data["le_location"]


def show_predict_page():

    st.title("UK Store Performance Predictor")

    st.write("""### Please provide below information to predict the performance of the store""")

    car_park = ["Yes", "No"]
    location = ["Shopping Centre", "Retail Park","High Street", "Village"]
    age = [age for age in range(1,16)]
    score = [score for score in range(10,21)]

    car_park.insert(0, "Select Car Park")
    location.insert(0, "Select Location")
    age.insert(0, "Select Store Age")
    score.insert(0, "Select Competition Score")

    car_park = st.selectbox("Car Park", car_park)
    location = st.selectbox("Location", location)
    age = st.selectbox("Store Age", age)
    score = st.selectbox("Competetion Score", score)


    ok = st.button("Show Performance Prediction")

    if ok:
        X = np.array([[car_park,location,age,score]])
        X[: ,0] = le_car.transform(X[: ,0])
        X[: ,1] = le_location.transform(X[:,1])
        X = X.astype(float)
        
        store_pred = dec_tree.predict(X)

        if store_pred[0] == 0:
            st.subheader("The store perforance is Bad ☹️")
        if store_pred[0] ==  1:
            st.subheader("The store perforance is Good 😃")