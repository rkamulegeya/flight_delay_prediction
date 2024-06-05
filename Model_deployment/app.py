from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import random
import os
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'CatB.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'train_data (1).csv')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f'Error loading model: {e}')
    model = None

df = pd.read_csv(DATA_PATH)

# Possible values for the constant features
id_values = df['id'].tolist()
departure_point_values = df['departure_point'].tolist()
arrival_point_values = df['arrival_point'].tolist()
scheduled_time_departure_values = df['scheduled_time_departure'].tolist()
scheduled_time_arrival_values = df['scheduled_time_arrival'].tolist()

# status_values = df['status'].tolist()
year_datop_values = df['year_datop'].tolist()
month_datop_values = df['month_datop'].tolist()
day_datop_values = df['day_datop'].tolist()
std_min_values = df['std_min'].tolist()

sta_hr_values = df['sta_hr'].tolist()
sta_min_values = df['sta_min'].tolist()
date_flightYear_values = df['date_flightYear'].tolist()
date_flightDay_values = df['date_flightDay'].tolist()
date_flightMonth_values = df['date_flightMonth'].tolist()

date_flightDayofweek_values = df['date_flightDayofweek'].tolist()
date_flightIs_month_end_values = df['date_flightIs_month_end'].tolist()
date_flightIs_month_start_values = df['date_flightIs_month_start'].tolist()
date_flightIs_quarter_end_values = df['date_flightIs_quarter_end'].tolist()

date_flightIs_quarter_start_values = df['date_flightIs_quarter_start'].tolist()
date_flightIs_year_end_values = df['date_flightIs_year_end'].tolist()
date_flightIs_year_start_values = df['date_flightIs_year_start'].tolist()
Season_values = df['Season'].tolist()
WeekofMonth_values = df['WeekofMonth'].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract features from the data
        date_flightDayofyear = int(data['date_flightDayofyear'])
        flight_number = data['flight_number']
        date_flightWeek = int(data['date_flightWeek'])
        aircraft_code = data['aircraft_code']
        std_hr = int(data['std_hr'])
        status=data['status']
        

        # Constant features - using random values
        id = random.choice(id_values)
        departure_point = random.choice(departure_point_values)
        arrival_point = random.choice(arrival_point_values)
        scheduled_time_departure = random.choice(scheduled_time_departure_values)
        scheduled_time_arrival = random.choice(scheduled_time_arrival_values)

        # status = random.choice(status_values)
        year_datop = random.choice(year_datop_values)
        month_datop = random.choice(month_datop_values)
        day_datop = random.choice(day_datop_values)
        std_min = random.choice(std_min_values)

        sta_hr = random.choice(sta_hr_values)
        sta_min = random.choice(sta_min_values)
        date_flightYear = random.choice(date_flightYear_values)
        date_flightDay = random.choice(date_flightDay_values)
        date_flightMonth = random.choice(date_flightMonth_values)

        date_flightDayofweek = random.choice(date_flightDayofweek_values)
        date_flightIs_month_end = random.choice(date_flightIs_month_end_values)
        date_flightIs_month_start = random.choice(date_flightIs_month_start_values)
        date_flightIs_quarter_end = random.choice(date_flightIs_quarter_end_values)
        
        date_flightIs_quarter_start = random.choice(date_flightIs_quarter_start_values)
        date_flightIs_year_end = random.choice(date_flightIs_year_end_values)
        date_flightIs_year_start = random.choice(date_flightIs_year_start_values)
        Season = random.choice(Season_values)
        WeekofMonth = random.choice(WeekofMonth_values)


        # Combine all features into a DataFrame
        input_data = pd.DataFrame({
            'id': [id],
            'flight_number': [flight_number],
            'departure_point': [departure_point],
            'arrival_point': [arrival_point],
            'scheduled_time_departure': [scheduled_time_departure],
            'scheduled_time_arrival': [scheduled_time_arrival],
            'status': [status],
            'aircraft_code': [aircraft_code],
            'year_datop': [year_datop],
            'month_datop': [month_datop],
            'day_datop': [day_datop],
            'std_hr': [std_hr],
            'std_min': [std_min],
            'sta_hr': [sta_hr],
            'sta_min': [sta_min],
            'date_flightYear': [date_flightYear],
            'date_flightMonth': [date_flightMonth],
            'date_flightWeek': [date_flightWeek],
            'date_flightDay': [date_flightDay],
            'date_flightDayofweek': [date_flightDayofweek],
            'date_flightDayofyear': [date_flightDayofyear],
            'date_flightIs_month_end': [date_flightIs_month_end],
            'date_flightIs_month_start': [date_flightIs_month_start],
            'date_flightIs_quarter_end': [date_flightIs_quarter_end],
            'date_flightIs_quarter_start': [date_flightIs_quarter_start],
            'date_flightIs_year_end': [date_flightIs_year_end],
            'date_flightIs_year_start': [date_flightIs_year_start],
            'Season': [Season],
            'WeekofMonth': [WeekofMonth]
        })

        # Make prediction
        prediction = model.predict(input_data)
        rounded_prediction = round(prediction[0], 0)
        return jsonify({'prediction': rounded_prediction})
    except Exception as e:
        print('Error in /predict endpoint:', e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)