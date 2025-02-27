import subprocess
import os
import sys

def run_command(command):
    """Run a shell command and check if it was successful."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing command: {e}")
        print(f"stderr: {e.stderr.decode()}")
        sys.exit(1)

print("Running Data Backup...")
run_command("python src/data/data_backup.py")

print("Running Data Processing...")
run_command("python src/data/data_processing.py")

print("Running Model Training...")
run_command("python src/model/model.py")

# Uncomment and update the below line if you need to run Model Prediction
# print("Running Model Prediction...")
# run_command("python src/model/predict.py")

# Get the port from environment variable or set to 8000 if not available
port = 8000
print(f"Starting FastAPI Server on port {port}...")

# Start FastAPI server without blocking
fastapi_process = subprocess.Popen(f"uvicorn src.backend.main:app --host 0.0.0.0 --port {port}", shell=True)

# Ensure FastAPI started properly (checking logs might help)
if fastapi_process.poll() is None:
    print("FastAPI server is running in the background.")

# Check if Render exposes Streamlit via a specific port (e.g., 8501)
streamlit_port = 8501
print(f"Launching Streamlit App on port {streamlit_port}...")

# Run Streamlit app
run_command(f"streamlit run src/frontend/app.py --server.port {streamlit_port}")
