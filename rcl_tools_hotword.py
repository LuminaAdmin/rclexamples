from typing import Any
import requests
from pathlib import Path
import json
import time
from enum import Enum
from requests import Response
import re
import os

#THIS SCRIPT MUST BE PROVIDED A SIMPLE TAB SEPERATED SENTENCE PAIR DATASET TO FUNCTION PROPERLY
#EACH SENTENCE MUST BE TERMINATED WITH PUNCUTATION
#EACH LINE MUST BE TERMINATED WITH A CARRIAGE RETURN

# Please set API_TOKEN to what is displayed from https://support.lumina247.com, please make sure the token is in the format "bearer <token>"
# "PASTE TOKEN HERE"
API_TOKEN = "PASTE TOKEN HERE"

# Default chunk size for uploads to 50MB
FILE_UPLOAD_CHUNK_SIZE = 50000000

# The vector size to use for training, best results tend to be between 3-5, you may need to adjust for best results
VECTOR_SIZE = 4

# You may change this line to point to a custom dataset for training
DATA_FOLDER = "rcl_dataset_hotword_sensor"

API_URL = "https://rclapi.lumina247.io"

HEADERS = {
    "Content-type": "application/json",
    "Accept": "application/json",
    "Authorization": API_TOKEN,
}

punct = {".", "?", "!"}
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

def clean_input(input):
    rem = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u2000-\u206F" #general punctuation
        u"\u2070-\u209F" #super and subscript
        u"\u20A0-\u20CF" #currency
        u"\u0300-\u036F" #combining diacritical marks
        u"\u20D0-\u20FF" #combining diacritical marks
        u"\u0378-\u0379" #certain greek and coptic
        u"\u2100-\u214F" #letterlike symbols
        u"\u2150-\u218F" #number forms
        u"\u2190-\u21FF" #arrows
        u"\u2200-\u22FF" #mathematical operators
        u"\u2300-\u23FF" #misc technical
        u"\u2400-\u243F" #control pictures
        u"\u2440-\u245F" #optical character recognition
        u"\u2460-\u24FF" #Enclosed Alphanumerics
        u"\u2500-\u257F" #box drawings
        u"\u2580-\u259F" #Block Elements
        u"\u25A0-\u25FF" #Geometric Shapes
        u"\u2600-\u26FF" #Miscellaneous Symbols
        u"\u2700-\u27BF" #Dingbats
        u"\u27C0-\u27EF" #Miscellaneous Mathematical Symbols-A
        u"\u27F0-\u27FF" #Supplemental Arrows-A
        u"\u2800-\u28FF" #Braille Patterns
        u"\u2900-\u297F" #Supplemental Arrows-B
        u"\u2980-\u29FF" #Miscellaneous Mathematical Symbols-B
        u"\u2A00-\u2AFF" #Supplemental Mathematical Operators
        u"\u2B00-\u2BFF" #Miscellaneous Symbols and Arrows
        u"\u2E80-\u2EFF" #CJK Radicals Supplement
        u"\u2F00-\u2FDF" #Kangxi Radicals
        u"\u2FF0-\u2FFF" #Ideographic Description Characters
        u"\u3000-\u303F" #CJK Symbols and Punctuation
        u"\u3040-\u309F" #Hiragana
        u"\u30A0-\u30FF" #Katakana
        u"\u3100-\u312F" #Bopomofo
        u"\u3130-\u318F" #Hangul Compatibility Jamo
        u"\u3190-\u319F" #Kanbun
        u"\u31A0-\u31BF" #Bopomofo Extended
        u"\u31F0-\u31FF" #Katakana Phonetic Extensions
        u"\u3200-\u32FF" #Enclosed CJK Letters and Months
        u"\u3300-\u33FF" #CJK Compatibility
        u"\u3400-\u4DBF" #CJK Unified Ideographs Extension A
        u"\u4DC0-\u4DFF" #Yijing Hexagram Symbols
        u"\u4E00-\u9FFF" #CJK Unified Ideographs
        u"\uA000-\uA48F" #Yi Syllables
        u"\uA490-\uA4CF" #Yi Radicals
        u"\uAC00-\uD7AF" #Hangul Syllables
        u"\uD800-\uDB7F" #High Surrogates
        u"\uDB80-\uDBFF" #High Private Use Surrogates
        u"\uDC00-\uDFFF" #Low Surrogates
        u"\uE000-\uF8FF" #Private Use Area
        u"\uF900-\uFAFF" #CJK Compatibility Ideographs
        u"\uFB00-\uFB4F" #Alphabetic Presentation Forms
        u"\uFB50-\uFDFF" #Arabic Presentation Forms-A
        u"\uFE00-\uFE0F" #Variation Selectors
        u"\uFE20-\uFE2F" #Combining Half Marks
        u"\uFE30-\uFE4F" #CJK Compatibility Forms
        u"\uFE50-\uFE6F" #Small Form Variants
        u"\uFE70-\uFEFF" #Arabic Presentation Forms-B
        u"\uFF00-\uFFEF" #Halfwidth and Fullwidth Forms
        u"\uFFF0-\uFFFF" #Specials
        u"\u02B0-\u02FF" #Spacing Modifier Letters
        u"\u0080-\u00BF" #certain special characters
        u"\U00010000-\U0001007F" #Linear B Syllabary
        u"\U00010080-\U000100FF" #Linear B Ideograms
        u"\U00010100-\U0001013F" #Aegean Numbers
        u"\U00010300-\U0001032F" #Old Italic
        u"\U00010330-\U0001034F" #Gothic
        u"\U00010380-\U0001039F" #Ugaritic
        u"\U00010400-\U0001044F" #Deseret
        u"\U00010450-\U0001047F" #Shavian
        u"\U00010480-\U000104AF" #Osmanya
        u"\U00010800-\U0001083F" #Cypriot Syllabary
        u"\U0001D000-\U0001D0FF" #Byzantine Musical Symbols
        u"\U0001D100-\U0001D1FF" #Musical Symbols
        u"\U0001D300-\U0001D35F" #Tai Xuan Jing Symbols
        u"\U0001D400-\U0001D7FF" #Mathematical Alphanumeric Symbols
        u"\U00020000-\U0002A6DF" #CJK Unified Ideographs Extension B
        u"\U0002F800-\U0002FA1F" #CJK Compatibility Ideographs Supplement
        u"\U000E0000-\U000E007F" #Tags                        
    "]+", re.UNICODE)

    return re.sub(rem, '', input)

def clean_dataset(path: str) -> str:

    def clean_line(line: str) -> str:
        lineBytes = line.encode('utf-8')
        cleansedLine =  clean_input(lineBytes.decode('utf-8')).strip()
        text = f"{cleansedLine}"        
        return text

    file = Path(path)
    new_file = file.parent / f"{file.stem}_cleaned{file.suffix}"
    lines = file.read_text(encoding="utf8").strip().split("\n")        
    text = "\n".join([clean_line(l) for l in lines])
    new_file.write_text(text, encoding="utf8")
    return new_file    

def parse_api_response(response: requests.Response) -> dict[str, Any]:
    """extracts json from response and raises errors when needed"""
    if response.status_code > 200:
        raise Exception("Error calling api")
    return {k.lower(): v for k, v in response.json().items()}

def create_session(description: str) -> int:
    """Creates an RCL session and returns the id"""
    endpoint = f"{API_URL}/trainingsession"
    r = requests.post(
        endpoint, json={"description": str(description)}, headers=HEADERS)
    return parse_api_response(r).get("trainingsessionkey", -1)

def get_session_info(session_key: int) -> dict[str, Any]:
    """Gets info about an rcl session"""
    r = requests.get(
        f"{API_URL}/trainingsession/{session_key}", headers=HEADERS)
    result = parse_api_response(r)
    try:
        return result
    except:
        raise Exception(f"Session {session_key} does not exist!")

def _multipart_upload_start(session_key: int, file_name: str) -> str:
    """Gets an id for a future file upload"""
    endpoint = f"{API_URL}/trainingsession/{session_key}/document/upload/multipart/{file_name}"
    
    print(f"Initializing Multipart upload")

    r = requests.post(endpoint, headers=HEADERS)
    result = parse_api_response(r)
    return result["documentid"]

def _multipart_upload_chunk(
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
    
    print(f"Finalizing Multipart upload")

    r = requests.post(
        endpoint,
        headers=HEADERS,
        json={},
    )
    return r.status_code == 200

def read_in_chunks(file_object, CHUNK_SIZE):
    
    print(f"Chunking upload, chunk_size: {CHUNK_SIZE}")

    while True:
        data = file_object.read(CHUNK_SIZE)
        if not data:
            break
        yield data
    
def upload_training_files(session_key: int, file: Path) -> bool:
    """Uploads a file by path"""    
    
    file_path = os.path.abspath(file)

    file_id = _multipart_upload_start(session_key, file.name)
   
    index = 0
    file_object = open(file_path, "rb")
    chunks = list(read_in_chunks(file_object, FILE_UPLOAD_CHUNK_SIZE))
    total_chunks = len(chunks)
    
    for chunk in chunks:
        try: 
            index = index + 1
            
            last_part = index == total_chunks
            
            print(f"Uploading chunk: {index} of {total_chunks}")
            
            ok = _multipart_upload_chunk(
                session_key,
                file_id,
                chunk,
                index,
                last_part
            )

            if not ok:
                return False

        except Exception as e:
            print(e)
            return False
    
    _multipart_upload_complete(
        session_key,
        file_id,
    )

    return True

def inference_host_ready_check(
    session_key: int,
    priority: InferencePriorityType = InferencePriorityType.index,
    detail: InferenceDetailType = InferenceDetailType.hot_word,
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

def train_model(
    session_key: int,
    vector_size: int,
    translation_model: bool = False,
    sensor_model: bool = False,
    train_goal: float = 0.7
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
    
    current_status = ""
    inference_ready_check = 0

    while(current_status != "trainingcompleted" and current_status != "trainingfailed"):
        info = get_session_info(session_key)
        current_status = info["trainingsession"]["statuses"][-1]["statusTypeName"].lower()
        print(f"Current Status: {current_status}")
        time.sleep(60)
    
    print (f"Health check complete, training status is {current_status}")

    while(inference_ready_check != 200):
        inference_ready_check = inference_host_ready_check(session_key, InferencePriorityType.index, InferenceDetailType.predict_next).status_code   
    
    if current_status == "trainingcompleted":
        return True
    else: 
        return False

def inference(
    session_key: int,
    input_text: str,
    priority: InferencePriorityType = InferencePriorityType.index,
    detail: InferenceDetailType = InferenceDetailType.hot_word,
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
    """End to end example for hotword model"""

    # UPDATE YOUR DESCRIPTION HERE
    session_key = create_session("Hotword Training")
    
    print(f"Created session: {session_key}")

    training_files = os.listdir(DATA_FOLDER)

    for file in training_files:            
        print(f"Cleaning file {file} before upload.")

        dataset = Path(f"{DATA_FOLDER}/").resolve().joinpath(file)

        clean_file = clean_dataset(dataset)

        upload_training_files(session_key, clean_file)

    print("Upload complete, starting training")

    training_successful = train_model(session_key, VECTOR_SIZE)

    if training_successful == False:
        print("Training did not complete successfully")
        return

    print("Training completed successfully, testing inference")
   
    sample = "a b d eeeee fffffff ggggg hhhhhhhhhh"

    print(f"Example input: {sample}")

    result = inference(
        session_key,
        sample,
        InferencePriorityType.index,
        InferenceDetailType.hot_word,
    )

    print("Example inference result:" + result)

if __name__ == "__main__":
    hotword_example()