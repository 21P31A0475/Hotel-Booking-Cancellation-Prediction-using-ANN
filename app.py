import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
df =pd.read_csv("hotel_bookings.csv")

model = load_model("model.keras")
preprocessor = joblib.load("ann_model.pkl")

st.title("Hotel Booking Cancellation Prediction")

st.image(
    "https://images.unsplash.com/photo-1439130490301-25e322d88054?q=80&w=1332&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    width=700
)

st.write("Enter booking details to predict whether the booking will be cancelled.")


hotel = st.selectbox("Hotel Type", df['hotel'].unique())

lead_time = st.number_input("Lead Time",min_value=int(df['lead_time'].min()),max_value=int(df['lead_time'].max()),value=50)
arrival_date_year = st.selectbox("Arrival Year", df['arrival_date_year'].unique())

arrival_date_month = st.selectbox("Month", [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
])

arrival_date_week_number = st.number_input("Week Number", 1, 53, 25)

stays_in_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
stays_in_week_nights = st.number_input("Week Nights", 0, 20, 2)

adults = st.number_input("Adults", 1, 10, 2)
children = st.number_input("Children", 0, 10, 0)
babies = st.number_input("Babies", 0, 5, 0)

meal = st.selectbox("Meal", df["meal"].unique())

market_segment = st.selectbox("Market Segment", df['market_segment'].unique())

distribution_channel = st.selectbox("Distribution Channel", df['distribution_channel'].unique())

is_repeated_guest = st.selectbox("Repeated Guest", df['is_repeated_guest'].unique())

previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
previous_bookings_not_canceled = st.number_input("Previous Non-Canceled", 0, 50, 0)

reserved_room_type = st.selectbox("Room Type", sorted(df['reserved_room_type'].unique()))

booking_changes = st.number_input("Booking Changes",min_value=0,max_value=10,value=0)

deposit_type = st.selectbox("Deposit Type", df['deposit_type'].unique())

days_in_waiting_list = st.number_input("Days in Waiting List",min_value=0,max_value=200,   value=0)

customer_type = st.selectbox("Customer Type", df['customer_type'].unique())

adr = st.number_input("ADR (Average Daily Rate)", 0.0, 500.0, 100.0)

required_car_parking_spaces = st.number_input("Parking Spaces", 0, 5, 0)

total_of_special_requests = st.number_input("Special Requests", 0, 10, 0)

country = st.selectbox("Country", df['country'].unique())


if st.button("Predict"):

    input_data = pd.DataFrame([{
        'hotel': hotel,
        'lead_time': lead_time,
        'arrival_date_year': arrival_date_year,
        'arrival_date_month': arrival_date_month,
        'arrival_date_week_number': arrival_date_week_number,
        'stays_in_weekend_nights': stays_in_weekend_nights,
        'stays_in_week_nights': stays_in_week_nights,
        'adults': adults,
        'children': children,
        'babies': babies,
        'meal': meal,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'is_repeated_guest': is_repeated_guest,
        'previous_cancellations': previous_cancellations,
        'previous_bookings_not_canceled': previous_bookings_not_canceled,
        'reserved_room_type': reserved_room_type,
        'booking_changes': booking_changes,
        'deposit_type': deposit_type,
        'days_in_waiting_list': days_in_waiting_list,
        'customer_type': customer_type,
        'adr': adr,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_of_special_requests': total_of_special_requests,
        'country': country
    }])

    input_transformed = preprocessor.transform(input_data)

    prob = model.predict(input_transformed)[0][0]

    if prob <= 0.5:
        st.success(f"Low chance of cancellation ({prob:.2f}%)")
    
    elif prob <= 0.75:
        st.warning(f"Moderate chance of cancellation ({prob:.2f}%)")

    else:
        st.error(f"High chance of cancellation ({prob:.2f}%)")