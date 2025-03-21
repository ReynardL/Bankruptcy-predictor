import requests
import time

url = "https://app-978501737888.us-central1.run.app/predict"
file_path = "Training/inference_test.csv"  

with open(file_path, 'rb') as f:
    files = {'file': f}
    start_time = time.time()
    
    try:
        response = requests.post(url, files=files)
        end_time = time.time()
        
        print(f"Time taken: {end_time - start_time:.4f} seconds")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}") 
