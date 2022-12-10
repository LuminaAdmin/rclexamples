from typing import Any
import requests
import json

# WARNING: STOPPING YOUR INFERENCE HOST WILL STOP YOUR ABILITY TO INFER FROM YOUR MODEL UNTIL IT IS STARTED BACK UP

# This example script demonstrates how to stop an inference host
# PLEASE SET THE FOLLOWING VARIABLES
# 1. API_TOKEN (retrieve from the support portal @ https://support.lumina247.com)
# 2. SET SESSION_KEY this is the session_key for your trained model

API_TOKEN = "bearer TOKEN_VALUE"

# SET YOUR TRAINING SESSION KEY HERE
SESSION_KEY = "trainingSessionKey"

API_URL = "https://rclapi.lumina247.io"

HEADERS = {
    "Content-type": "application/json",
    "Accept": "application/json",
    "Authorization": API_TOKEN,
}

def stop_host():
    endpoint = (
        f"{API_URL}"
        f"/inferencehost/{SESSION_KEY}"
        f"/stop"        
    )

    r = requests.post(endpoint, data=json.dumps(SESSION_KEY), headers=HEADERS)
    
    if r.status_code != 200:
        raise Exception(f"Error calling inference: {r.json()}")
    else:
        print('INFERENCE HOST STOPPING')

if __name__ == "__main__":
    stop_host()