# Start tensorflow serving
docker run -p 8501:8501 \
     --mount type=bind,source="$(pwd)"/trainer/predict_spend_model_a/export/predict_spend_a,target=/models/predict_spend_a \
     --mount type=bind,source="$(pwd)"/trainer/predict_spend_model_b/export/predict_spend_b,target=/models/predict_spend_b \
     --mount type=bind,source="$(pwd)"/trainer/predict_activity_change_model_a/export/predict_activity_change_a,target=/models/predict_activity_change_a \
      --mount type=bind,source="$(pwd)"/trainer/predict_activity_change_model_b/export/predict_activity_change_b,target=/models/predict_activity_change_b \
     --mount type=bind,source="$(pwd)"/trainer/models.config,target=/models/models.config \
     -t tensorflow/serving --model_config_file=/models/models.config &

# Start flask api server
# FLASK_APP=api/group_allocator_api.py flask run --host=0.0.0.0 &
