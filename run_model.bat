@echo off
echo Activating Virtual Environment...
call venv\Scripts\activate

echo Running Data Procesing...
python src\data\data_processing.py

echo Running Model Training...
python src\model\model.py

echo Running Model Prediction...
python src\model\predict.py

echo Starting FastAPI Server...
start cmd /k "python src\backend\main.py"

echo Launching Streamlit App...
start cmd /k "streamlit run src\frontend\app.py"

echo All services started successfully!
