# Start tensorflow serving
docker run -p 8501:8501 \
     --mount type=bind,source="$(pwd)"/trainer/predict_spend_model/export/predict_spend,target=/models/predict_spend \
     --mount type=bind,source="$(pwd)"/trainer/predict_activity_change_model/export/predict_activity_change,target=/models/predict_activity_change \
     --mount type=bind,source="$(pwd)"/trainer/models.config,target=/models/models.config \
     -t tensorflow/serving --model_config_file=/models/models.config &

# Start flask api server
FLASK_APP=api/group_allocator_api.py flask run --host=0.0.0.0 &
