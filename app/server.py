import flask
from flask import request, jsonify
from pathlib import Path
from marshmallow import Schema, fields, ValidationError
from session import create_model_for_provider


app = flask.Flask(__name__)
onnx_model = create_model_for_provider(Path('/onnx_server/model.onnx'))


class InputSchema(Schema):
    pass


input_schema = InputSchema(many=True)


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"message": "No input data provided"}), 400
    
    instances = json_data.get('instances', None)
    if not instances:
        return jsonify({"message": "No instances provided"}), 400
    
    try:
        instances = input_schema.load(instances)
    except ValidationError as err:
        return jsonify({"message": str(err.messages)}), 422

    
    return jsonify({'predictions': onnx_model.predict(instances)})


@app.route('/health_check', methods=['GET'])
def health_check():
    return jsonify({'is_healthy': True}), 200


if __name__ == "__main__":
    app.run()
