""" Api endpoints """
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

import api_helper

api_helper.set_config()

app = Flask(__name__)
CORS(app)

@app.route('/group_allocator/predict', methods=['POST'])
def group_allocator():
    is_valid_input, message, inputs = api_helper.validate_input(request)

    if not is_valid_input:
        return jsonify(success=False, errors = { 'message': message })

    group, spend_result, activity_change = api_helper.get_allocated_group(inputs, request)

    return jsonify({ 'result': {
        'group': group,
        'spendResult': spend_result,
        'activityChange': activity_change
    } })
