from flask import Flask, render_template, request, jsonify
from src.components.prediction import Predictor, Features


app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"message": "Financial Inclusion Africa API"})


@app.route('/predict', methods=['POST'])
def predict():
    data_obj = Features(
        location_type=request.json['location_type'],
        cellphone_access=request.json['cellphone_access'],
        household_size=request.json['household_size'],
        age=request.json['age'],
        gender=request.json['gender'],
        relationship_with_head=request.json['relationship_with_head'],
        marital_status=request.json['marital_status'],
        education_level=request.json['education_level'],
        job_type=request.json['job_type']
    )

    # print(request.json)

    data_df = data_obj.convert_to_dataframe()
    predictor = Predictor()

    pred1, pred2, prob1, prob2 = predictor.predict(data_df)

    return jsonify({
        "message": "success",
        "data": {
            "base_model": {
                "prediction": pred1,
                "probablity": prob1,
            },
            "opt_model": {
                "prediction": pred2,
                "probablity": prob2,
            }
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
