#!/bin/bash

# Activating Virtual Environment (if needed)
# Uncomment the line below if you need to activate the virtual environment.
# source venv1/bin/activate

echo "Running Data Backup..."
python src/data/data_backup.py

echo "Running Data Processing..."
python src/data/data_processing.py

echo "Running Model Training..."
python src/model/model.py

# Uncomment and update the below line if you need to run Model Prediction
# echo "Running Model Prediction..."
# python src/model/predict.py

echo "Starting FastAPI Server..."
# Running FastAPI with uvicorn
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 &  # The '&' runs the process in the background.

# Uncomment and update the below line if you need to run Streamlit
echo "Launching Streamlit App..."
streamlit run src/frontend/app.py &

echo "All services started successfully!"
