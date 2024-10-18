from flask import Flask, request, jsonify, json, make_response
from flask_cors import CORS, cross_origin
from werkzeug.exceptions import HTTPException

from src.components.prediction import Predictor, Features


# Instantiate API server
app = Flask(__name__)

# Enable Cross Origin Resource sharing for server
# CORS(app, supports_credentials=True)


# API Routes
@app.route("/")
def index():
    return jsonify({"message": "Financial Inclusion Africa API"})


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    data_obj = Features(
        location_type=request.json["locationType"],
        cellphone_access=request.json["cellphoneAccess"],
        household_size=request.json["householdSize"],
        age=request.json["age"],
        gender=request.json["gender"],
        relationship_with_head=request.json["relationshipWithHead"],
        marital_status=request.json["maritalStatus"],
        education_level=request.json["educationLevel"],
        job_type=request.json["jobType"],
    )

    # print(request.json)

    data_df = data_obj.convert_to_dataframe()
    predictor = Predictor()

    pred1, pred2, prob1, prob2 = predictor.predict(data_df)

    return jsonify(
        {
            "message": "success",
            "data": {
                "base_model": {
                    "prediction": pred1,
                    "probablity": prob1,
                },
                "opt_model": {
                    "prediction": pred2,
                    "probablity": prob2,
                },
            },
        }
    )


# catch every type of exception
@app.errorhandler(Exception)
def handle_exception(e):
    # print(e)

    # Error is a HTTP Exception
    if isinstance(e, HTTPException):
        response = e.get_response()

        # response = json.dumps({ "error": e.description })
        # response["message"] = e.description  # type: ignore
        # response["status_code"] = e.code  # type: ignore
        setattr(response, "message", e.description)
        setattr(response, "status", e.code)

    else:
        # build response
        response = make_response(jsonify({"message": "Something went wrong"}), 500)

    # Add the CORS header
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.content_type = "application/json"

    return response


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(port=8000)
