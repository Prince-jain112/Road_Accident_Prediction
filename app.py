
from flask import Flask, render_template, request, redirect
from flask import Flask, render_template, request
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

app = Flask(__name__)

def init_db():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            state TEXT,
            weather TEXT,
            visibility TEXT,
            traffic_density TEXT,
            temperature INTEGER,
            vehicles INTEGER,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]

    if "created_at" not in columns:
        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN created_at TIMESTAMP
    """)

    conn.commit()
    conn.close()



# Load AI Model
model = joblib.load("random_forest_model.pkl")

# Load Encoders
encoders = joblib.load("encoders.pkl")

print("✅ Model Loaded")
print("✅ Encoders Loaded")

print("\nSupported Cities:")
print(encoders["city"].classes_)


@app.route("/")
def home():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    # Prediction History
    cursor.execute("""
        SELECT id, city, state, weather, visibility,
               traffic_density, temperature, vehicles,
               prediction, confidence, created_at
        FROM predictions
        ORDER BY id DESC
    """)

    history = cursor.fetchall()

    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    # Accident Severity Counts
    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Fatal Accident'
    """)
    fatal_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Major Accident'
    """)
    major_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Minor Accident'
    """)
    minor_count = cursor.fetchone()[0]

    # Percentages
    fatal_percentage = round(
        (fatal_count / total_predictions) * 100, 2
    ) if total_predictions > 0 else 0

    major_percentage = round(
        (major_count / total_predictions) * 100, 2
    ) if total_predictions > 0 else 0

    minor_percentage = round(
        (minor_count / total_predictions) * 100, 2
    ) if total_predictions > 0 else 0

    # City-wise Statistics
    cursor.execute("""
        SELECT city, COUNT(*)
        FROM predictions
        GROUP BY city
        ORDER BY COUNT(*) DESC
    """)

    city_statistics = cursor.fetchall()
    print("CITY STATISTICS:", city_statistics)

    # Severity-wise Statistics
    cursor.execute("""
        SELECT prediction, COUNT(*)
        FROM predictions
        GROUP BY prediction
    """)

    severity_statistics = cursor.fetchall()

    print("SEVERITY STATISTICS:", severity_statistics)
    # Severity-wise City Statistics
    cursor.execute("""
        SELECT city, prediction, COUNT(*)
        FROM predictions
        GROUP BY city, prediction
        ORDER BY city
    """)

    severity_data = cursor.fetchall()

    print("SEVERITY DATA:", severity_data)



    conn.close()

    return render_template(
        "index.html",

        history=history,

        total_predictions=total_predictions,

        fatal_count=fatal_count,
        major_count=major_count,
        minor_count=minor_count,

        fatal_percentage=fatal_percentage,
        major_percentage=major_percentage,
        minor_percentage=minor_percentage,

        city_statistics=city_statistics,
        severity_statistics=severity_statistics,
        severity_data=severity_data
    )


@app.route("/predict", methods=["POST"])
def predict():
    city = request.form["city"]
    state = request.form["state"]
    road_type = request.form["road_type"]
    weather = request.form["weather"]
    visibility = request.form["visibility"]
    traffic_density = request.form["traffic_density"]
    temperature = int(request.form["temperature"])
    hour = int(request.form["hour"])
    day_of_week = request.form["day_of_week"]
    is_weekend = int(request.form["is_weekend"])
    traffic_signal = int(request.form["traffic_signal"])
    vehicles_involved = int(request.form["vehicles_involved"])
    is_peak_hour = int(request.form["is_peak_hour"])

  

    
    print("\n========== USER INPUT ==========")

    print("City :", city)
    print("State :", state)
    print("Road Type :", road_type)
    print("Weather :", weather)
    print("Visibility :", visibility)
    print("Traffic Density :", traffic_density)
    print("Temperature :", temperature)
    print("Hour :", hour)
    print("Day :", day_of_week)
    print("Weekend :", is_weekend)
    print("Traffic Signal :", traffic_signal)
    print("Vehicles :", vehicles_involved)
    print("Peak Hour :", is_peak_hour)



    original_city = city
    original_state = state
    original_weather = weather
    original_visibility = visibility
    original_traffic_density = traffic_density
    # Encode Text Values

    city = encoders["city"].transform([city])[0]
    state = encoders["state"].transform([state])[0]
    road_type = encoders["road_type"].transform([road_type])[0]
    weather = encoders["weather"].transform([weather])[0]
    visibility = encoders["visibility"].transform([visibility])[0]
    traffic_density = encoders["traffic_density"].transform([traffic_density])[0]
    day_of_week = encoders["day_of_week"].transform([day_of_week])[0]

   
    
    print("\n===== ENCODED VALUES =====")

    print("City :", city)
    print("State :", state)
    print("Road Type :", road_type)
    print("Weather :", weather)
    print("Visibility :", visibility)
    print("Traffic Density :", traffic_density)
    print("Day :", day_of_week)

     # Prepare Input for AI Model
    input_data = pd.DataFrame([[
    city,
    state,
    road_type,
    weather,
    visibility,
    traffic_density,
    temperature,
    hour,
    day_of_week,
    is_weekend,
    traffic_signal,
    vehicles_involved,
    is_peak_hour
]], columns=[
    "city",
    "state",
    "road_type",
    "weather",
    "visibility",
    "traffic_density",
    "temperature",
    "hour",
    "day_of_week",
    "is_weekend",
    "traffic_signal",
    "vehicles_involved",
    "is_peak_hour"
])
    
    # AI Prediction
    prediction = model.predict(input_data)
    
    print("\n===== AI PREDICTION =====")
    print("Prediction :", prediction)
    prediction = prediction[0]
    probabilities = model.predict_proba(input_data)[0]

    risk_probability = probabilities[prediction] * 100

    if prediction == 0:
        result = "Fatal Accident"

    elif prediction == 1:
        result = "Major Accident"

    else:
        result = "Minor Accident"

    risk_probability = round(risk_probability, 2)

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            city,
            state,
            weather,
            visibility,
            traffic_density,
            temperature,
            vehicles,
            prediction,
            confidence,
            created_at
        )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    original_city,
    original_state,
    original_weather,
    original_visibility,    
    original_traffic_density,
    temperature,
    vehicles_involved,
    result,
    risk_probability,
    datetime.now()
))

    conn.commit()
    conn.close()


    # Safety Advice

   # Safety Advice

    if prediction == 0:
        advice = "⚠️ High Risk! Drive carefully, reduce speed and maintain a safe distance."

    elif prediction == 1:
        advice = "⚠️ Moderate Risk! Follow traffic rules and avoid risky driving."

    else:
        advice = "✅ Low Risk! Continue following safe driving practices."


# Additional Safety Recommendations

    recommendations = []

    if original_visibility == "low":
        recommendations.append(
            "👁️ Low visibility detected. Use headlights and reduce speed."
        )

    if original_traffic_density == "high":
        recommendations.append(
            "🚦 High traffic density detected. Avoid sudden lane changes."
        )

    if vehicles_involved >= 4:
        recommendations.append(
            "🚗 Multiple vehicles involved. Maintain extra safety distance."
        )

    if temperature >= 40:
        recommendations.append(
            "🌡️ High temperature detected. Stay alert and take breaks if needed."
        )

    if is_peak_hour == 1:
        recommendations.append(
            "⏰ Peak hour detected. Expect heavier traffic and drive patiently."
        )

    if not recommendations:
        recommendations.append(
            "✅ No additional risk factors detected. Continue following safe driving practices."
        )


    
   

    # Get prediction history
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, city, state, weather, visibility,
            traffic_density, temperature, vehicles,
            prediction, confidence, created_at
        FROM predictions
        ORDER BY id DESC
    """)

    history = cursor.fetchall()

    conn.close()


        # Prediction Statistics

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Fatal Accident'
    """)
    fatal_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Major Accident'
    """)
    major_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Minor Accident'
    """)
    minor_count = cursor.fetchone()[0]

    fatal_percentage = round((fatal_count / total_predictions) * 100, 2) if total_predictions > 0 else 0
    major_percentage = round((major_count / total_predictions) * 100, 2) if total_predictions > 0 else 0
    minor_percentage = round((minor_count / total_predictions) * 100, 2) if total_predictions > 0 else 0

    conn.close()

    # City-wise Prediction Statistics
    # City-wise Prediction Statistics
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT city, COUNT(*)
        FROM predictions
        GROUP BY city
        ORDER BY COUNT(*) DESC
    """)

    city_statistics = cursor.fetchall()

    # Severity-wise Statistics
    cursor.execute("""
        SELECT prediction, COUNT(*)
        FROM predictions
        GROUP BY prediction
    """)

    severity_statistics = cursor.fetchall()

    print("SEVERITY STATISTICS:", severity_statistics)

    conn.close()

    print("CITY STATISTICS:", city_statistics)

    # Severity-wise City Statistics
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT city, prediction, COUNT(*)
        FROM predictions
        GROUP BY city, prediction
        ORDER BY city
    """)

    severity_data = cursor.fetchall()

    

    print("SEVERITY DATA:", severity_data)
    conn.close()

    return render_template(
        "index.html",
        prediction=result,
        advice=advice,
        risk_probability=risk_probability,
        history=history,
        recommendations=recommendations,
        total_predictions=total_predictions,
        fatal_count=fatal_count,
        major_count=major_count,
        minor_count=minor_count,
        fatal_percentage=fatal_percentage,
        major_percentage=major_percentage,
        minor_percentage=minor_percentage,
        city_statistics=city_statistics,
        severity_data=severity_data,
        city=original_city,
        state=original_state,
        weather=original_weather,
        visibility=original_visibility,
        vehicles_involved=vehicles_involved,
        severity_statistics=severity_statistics,
        temperature=temperature
    )

@app.route("/clear-history", methods=["POST"])
def clear_history():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()

    return redirect("/")

@app.route("/delete-prediction/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE id = ?",
        (prediction_id,)
    )

    conn.commit()
    conn.close()

    return redirect("/")

if __name__ == "__main__":
    init_db()
    app.run(debug=False)