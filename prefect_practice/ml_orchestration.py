import time
import os
from pathlib import Path
from datetime import timedelta
from prefect import flow, task

TEMP_DIR = Path("temp_ml_data")
DUMMY_FILE_PATH = TEMP_DIR / "dummy_data.txt"

@task
def download_data(file_path: Path):
    print(f"Task A: Downloading data to {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write("This is some simulated data.\n")
        f.write("Line 2 of data.\n")
        f.write("Line 3 of data.\n")
    print(f"Task A: Data downloaded successfully to {file_path}")
    time.sleep(20)
    return str(file_path)

@task
def preprocess_data(data_file_path: str):
    print(f"Task B: Preprocessing data from {data_file_path}...")
    try:
        with open(data_file_path, "r") as f:
            content = f.read()
            print("--- Dummy File Content ---")
            print(content)
            print("--- End of Content ---")
        print("Task B: Data preprocessed successfully.")
        time.sleep(15)
        return "preprocessed_data_summary"
    except FileNotFoundError:
        print(f"Error: File not found at {data_file_path}")
        raise

@task
def train_model(preprocessed_summary: str):
    print(f"Task C: Training model with {preprocessed_summary}...")
    print("Task C: Model training complete.")
    time.sleep(15)
    return "trained_model_v1.0"

@flow(name="ML Orchestration Example Flow")
def ml_orchestration_workflow():
    print("\nStarting ML Orchestration Workflow...")
    data_path_result = download_data(file_path=DUMMY_FILE_PATH)
    preprocessed_result = preprocess_data(data_file_path=data_path_result)
    trained_model_name = train_model(preprocessed_summary=preprocessed_result)
    print(f"\nWorkflow finished. Final output: {trained_model_name}")

    if DUMMY_FILE_PATH.exists():
        os.remove(DUMMY_FILE_PATH)
        print(f"Cleaned up dummy file: {DUMMY_FILE_PATH}")
    if TEMP_DIR.exists() and not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)
        print(f"Cleaned up dummy directory: {TEMP_DIR}")

if __name__ == "__main__":
    ml_orchestration_workflow()


# configure the api with prefect app.prefect.cloud --> prefect cloud login -k YOUR_API_KEY_HERE
# run python ----.py

