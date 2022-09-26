from operator import truediv
from typing import Any
import requests
from pathlib import Path
import json
import time
from enum import Enum
from requests import Response

# Please set API_TOKEN to what is displayed from https://support.lumina247.com, please make sure the token is in the format "***REMOVED***<token>"
# "PASTE TOKEN HERE"
API_TOKEN = "PASTE TOKEN HERE"

# You may change this line to point to a custom dataset for training
DATA_SET = "rcl_dataset_hotword_sensor/_reinforcement_learning.txt"

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


def create_session(description: str) -> int:
    """Creates an RCL session and returns the id"""
    endpoint = f"{API_URL}/trainingsession"
    r = requests.post(endpoint, json={"description": str(description)}, headers=HEADERS)
    return parse_api_response(r).get("trainingsessionkey", -1)


def get_session_info(session_key: int) -> dict[str, Any]:
    """Gets info about an rcl session"""
    r = requests.get(f"{API_URL}/trainingsession/{session_key}", headers=HEADERS)
    result = parse_api_response(r)
    try:
        return result
    except:
        raise Exception(f"Session {session_key} does not exist!")


def upload_file(session_key: int, file: Path) -> bool:
    """Uploads a file by path"""

    def _multipart_upload_start(session_key: int, file_name: str) -> str:
        """Gets an id for a future file upload"""
        endpoint = f"{API_URL}/trainingsession/{session_key}/document/upload/multipart/{file_name}"
        r = requests.post(endpoint, headers=HEADERS)
        result = parse_api_response(r)
        return result["documentid"]

    def _multipart_upload_do(
        session_key: int,
        document_id: str,
        content: bytes,
        part_number: int = 1,
        last_part: bool = True,
    ) -> bool:
        """Uploads a part of a multipart file"""
        endpoint = (
            f"{API_URL}"
            f"/trainingsession/{session_key}"
            f"/document/{document_id}"
            f"/upload/multipart/{part_number}/{last_part}"
        )
        r = requests.put(
            endpoint,
            data=content,
            headers={
                "Content-Type": "application/octet-stream",
                "Authorization": API_TOKEN,
            },
        )
        return r.status_code == 200

    def _multipart_upload_complete(session_key: int, document_id: str) -> bool:
        """Marks a multipart upload as complete"""
        endpoint = (
            f"{API_URL}"
            f"/trainingsession/{session_key}"
            f"/document/{document_id}/upload/multipart/complete"
        )
        r = requests.post(
            endpoint,
            headers=HEADERS,
            json={},
        )
        return r.status_code == 200

    file_id = _multipart_upload_start(session_key, file.name)
    ok = _multipart_upload_do(
        session_key,
        file_id,
        file.read_bytes(),
    )
    if ok:
        complete = _multipart_upload_complete(
            session_key,
            file_id,
        )
        return complete
    return False


def training_ready_check(
    session_key: int,
    priority: InferencePriorityType = InferencePriorityType.accuracy,
    detail: InferenceDetailType = InferenceDetailType.predict_next,
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
    return r


def train_model(
    session_key: int,
    vector_size: int,
    translation_model: bool,
    sensor_model: bool,
    train_goal: float,
    block_for_ready: bool,
) -> bool:
    """Trains the RCL Model with the provided settings"""
    r = requests.post(
        f"{API_URL}/trainingsession/{session_key}/start",
        json={
            "vectorSize": vector_size,
            "trainTranslationServices": translation_model,
            "trainSensorServices": sensor_model,
            "trainGoal": train_goal,
        },
        headers=HEADERS,
    )

    if r.status_code != 200:
        raise Exception("Training Error!")

    if block_for_ready:
        info = get_session_info(session_key)
        health_check_count = 0
        # minimum successful healthy inferences before "ready"
        health_check_threshold = 5

        while (
            info["trainingsession"]["statuses"][-1]["statusTypeName"].lower()
            != "trainingcompleted"
        ):
            time.sleep(5)
            info = get_session_info(session_key)

        health_status = training_ready_check(
            session_key, InferencePriorityType.index, InferenceDetailType.hot_word
        )
        while health_check_count < health_check_threshold:
            health_status = training_ready_check(
                session_key, InferencePriorityType.index, InferenceDetailType.hot_word
            )
            if health_status.status_code == 200:
                response_body = health_status.json()
                if "unrecognized model" not in response_body:
                    health_check_count += 1

            if health_status != 200:
                time.sleep(5)

        print("Model is ready for inference.")


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


def hotword_example():
    # UPDATE YOUR DESCRIPTION HERE
    session_key = create_session("Self Driving Car Sensor Data")
    print(f"Created session: {session_key}")

    print("uploading files")
    dataset = Path(DATA_SET)
    upload_file(session_key, dataset)

    print("Training Sensor Input Model - HotWord Inference with Train Goal to 70%")

    train_model(
        session_key=session_key,
        vector_size=5,
        translation_model=False,
        sensor_model=True,
        train_goal=0.7,
        block_for_ready=True,
    )

    print("Training done!")
    print("Testing inference: hotword")

    # input example 1
    example1 = "a b d eeeee fffffff ggggg hhhhhhhhhh"
    print("Example1 inference input:" + example1)

    # input example 2
    example2 = "a b c dd hhhhhhhhhh"
    print("Example2 inference input:" + example2)

    # output example 1
    result1 = inference(
        session_key,
        example1,
        InferencePriorityType.index,
        InferenceDetailType.hot_word,
    )
    print("Example1 inference result:" + result1)

    # output example 2
    result2 = inference(
        session_key,
        example2,
        InferencePriorityType.index,
        InferenceDetailType.hot_word,
    )
    print("Example2 inference result:" + result2)


if __name__ == "__main__":
    hotword_example()
