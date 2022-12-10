from typing import Any
import requests
import json
from enum import Enum
from requests import Response
import time

# This example script demonstrates how to infer from a trained model
# The service will automatically start a new host for inference, wait for the host to be ready and then execute an example inference request.
# You will want to extend this example for multiple inference or to programatically include in your own applications
# PLEASE SET THE FOLLOWING VARIABLES
# 1. API_TOKEN (retrieve from the support portal @ https://support.lumina247.com)
# 2. SET SESSION_KEY this is the session_key for your trained model
# 3. SET VECTOR_SIZE this is the vecrot size you used for training

API_TOKEN = "bearer TOKEN_VALUE"

# SET YOUR TRAINING SESSION KEY HERE
SESSION_KEY = "trainingSessionKey"

# The vector size to use for training, best results tend to be between 3-5, you may need to adjust for best results
VECTOR_SIZE = 5

API_URL = "https://rclapi.lumina247.io"

HEADERS = {
    "Content-type": "application/json",
    "Accept": "application/json",
    "Authorization": API_TOKEN,
}

class InferenceDetailType(Enum):
    """Enums containing the id's for inference types"""
    search = "Search"
    predict_line = "PredictLine"
    predict_next = "PredictNext"
    translate_line = "TranslateLine"
    hot_word = "HotWord"

class InferencePriorityType(Enum):
    """Enums containing the id's for training optimizers"""
    index = "Index"
    accuracy = "Accuracy"
    specific = "Specific"

def parse_api_response(response: requests.Response) -> dict[str, Any]:
    """extracts json from response and raises errors when needed"""
    if response.status_code > 200:
        raise Exception("Error calling api")
    return {k.lower(): v for k, v in response.json().items()}

def get_session_info(session_key: int) -> dict[str, Any]:
    """Gets info about an rcl session"""
    r = requests.get(
        f"{API_URL}/trainingsession/{session_key}", headers=HEADERS)
    result = parse_api_response(r)
    try:
        return result
    except:
        raise Exception(f"Session {session_key} does not exist!")
 
def inference_host_ready_check(
    session_key: int,
    priority: InferencePriorityType = InferencePriorityType.index,
    detail: InferenceDetailType = InferenceDetailType.translate_line,
) -> Response:
    """Makes an inference with the model"""

    print("Checking if model is ready for inference.")

    endpoint = (
        f"{API_URL}"
        f"/trainingsession/{session_key}"
        f"/inference/{priority.value}"
        f"/{detail.value}"
    )
    test_data = "health check"
    r = requests.post(endpoint, data=json.dumps(test_data), headers=HEADERS)
    
    if "unrecognized model" in str(r.content):
        return inference_host_ready_check(session_key, priority, detail)

    return r

def translate_inference_example(
    session_key: int,
    input_text: str,
    priority: InferencePriorityType = InferencePriorityType.index,
    detail: InferenceDetailType = InferenceDetailType.translate_line,
) -> str:
    """Makes an inference with the model"""
    endpoint = (
        f"{API_URL}"
        f"/trainingsession/{session_key}"
        f"/inference/{priority.value}"
        f"/{detail.value}"
    )
    r = requests.post(endpoint, data=json.dumps(input_text), headers=HEADERS)
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception(f"Error calling inference: {r.json()}")

def inference(
    session_key: int,
    input_text: str,
    priority: InferencePriorityType = InferencePriorityType.accuracy,
    detail: InferenceDetailType = InferenceDetailType.predict_next,
) -> str:
    """Makes an inference with the model"""
    endpoint = (
        f"{API_URL}"
        f"/trainingsession/{session_key}"
        f"/inference/{priority.value}"
        f"/{detail.value}"
    )
    r = requests.post(endpoint, data=json.dumps(input_text), headers=HEADERS)
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception(f"Error calling inference: {r.json()}")

def start_host():
    endpoint = (
        f"{API_URL}"
        f"/inferencehost/{SESSION_KEY}"
        f"/start"        
    )

    r = requests.post(endpoint, data=json.dumps(SESSION_KEY), headers=HEADERS)
    
    if r.status_code != 200:
        raise Exception(f"Error calling inference: {r.json()}")
    else:
        print('INFERENCE HOST STARTING')
    
    inference_ready_check = 0

    while(inference_ready_check != 200):
        inference_ready_check = inference_host_ready_check(SESSION_KEY, InferencePriorityType.index, InferenceDetailType.translate_line).status_code
        time.sleep(10)

def translation_example():
    start_host()

    example_inference = "Tom fell in love"

    inference_result = translate_inference_example(SESSION_KEY, example_inference)

    print(f"Inference Result: {inference_result['content']}")

if __name__ == "__main__":
    translation_example()