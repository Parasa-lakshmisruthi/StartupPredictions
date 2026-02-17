from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load trained model
model = joblib.load("random_forest_model.pkl")

# home page
@app.route('/')
def home():
    return render_template('index.html')




# prediction route
@app.route('/predict', methods=['POST'])
def predict():

    age_first_funding_year = float(request.form['age_first_funding_year'])
    age_last_funding_year = float(request.form['age_last_funding_year'])
    age_first_milestone_year = float(request.form['age_first_milestone_year'])
    age_last_milestone_year = float(request.form['age_last_milestone_year'])
    relationships = float(request.form['relationships'])
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    avg_participants = float(request.form['avg_participants'])

    input_data = np.array([[age_first_funding_year,
                            age_last_funding_year,
                            age_first_milestone_year,
                            age_last_milestone_year,
                            relationships,
                            funding_rounds,
                            funding_total_usd,
                            milestones,
                            avg_participants]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Acquired"
    else:
        result = "Closed"

    return render_template('result.html', result=result)





if __name__ == "__main__":
    app.run(debug=True, port=5050)

