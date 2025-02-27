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

# Get the port from Render environment variable or set to 8000 if not available
PORT=${PORT:-8000}

echo "Starting FastAPI Server on port $PORT..."
# Running FastAPI with uvicorn
uvicorn src.backend.main:app --host 0.0.0.0 --port $PORT &

# Check if FastAPI started properly
if [ $? -eq 0 ]; then
    echo "FastAPI Server started successfully."
else
    echo "Failed to start FastAPI Server."
    exit 1
fi

# Check if Render exposes Streamlit via a specific port (e.g., 8501)
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

echo "Launching Streamlit App on port $STREAMLIT_PORT..."
# Running Streamlit App
streamlit run src/frontend/app.py --server.port $STREAMLIT_PORT &

# Check if Streamlit started properly
if [ $? -eq 0 ]; then
    echo "Streamlit App launched successfully."
else
    echo "Failed to launch Streamlit App."
    exit 1
fi

echo "All services started successfully!"
