from typing import Any
import requests
from pathlib import Path
import random
import re
import os
import time

# THIS SCRIPT TRAINS A NEW MODEL FOR TRANSLATION
# PLEASE SET THE FOLLOWING VARIABLES
# 1. API_TOKEN (retrieve from the support portal @ https://support.lumina247.com)
# 2. SET SESSION_KEY this is the session_key for your trained model
# 3. SET VECTOR_SIZE this is the vecrot size you used for training
# 4. SET TEST_PATH this is a path to your test data

# "PASTE TOKEN HERE"
API_TOKEN = "bearer TOKEN_VALUE"

# Default chunk size for uploads to 50MB
FILE_UPLOAD_CHUNK_SIZE = 50000000

# The vector size to use for training, best results tend to be between 3-5, you may need to adjust for best results
VECTOR_SIZE = 5

# You may change this line to point to a custom dataset for training
DATA_FOLDER = "rcl_dataset_translate"

API_URL = "https://rclapi.lumina247.io"

HEADERS = {
    "Content-type": "application/json",
    "Accept": "application/json",
    "Authorization": API_TOKEN,
}

punct = {".", "?", "!"}

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
    """Cleans a Tab seperated dataset for training purposes"""
    def clean_line(line: str) -> str:
        parts = line.split("\t")
        if len(parts) > 1:
            p1 = parts[0].strip()
            if p1[-1] not in punct:
                p1 = f"{p1}."
        
            p2 = parts[1].strip()
            if p2[-1] not in punct:
                p2 = f"{p2}."
        
            part1bytes = p1.encode('utf-8')
            part2bytes = p2.encode('utf-8')

            part1 = clean_input(part1bytes.decode('utf-8')).strip()
            part2 = clean_input(part2bytes.decode('utf-8')).strip()

            text = f"{part1}\t{part2}"

            return text
        return ""

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

def sample_translation_set(path: Path, sample_size: int = 100, vector_size: int = 5):
    # translate >= vector length
    keep = []
    lines = path.read_text(encoding="utf8").strip().split("\n")
    for l in lines:
        en, xx = l.split("\t")
        en, xx = (en.strip(), xx.strip())
        if len(en.split(" ")) >= vector_size:
            keep.append((en, xx))

    random.shuffle(keep)
    sample = keep[:sample_size]
    return sample
    
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

def check_training_failed(session_key: int):
    r = requests.get(f"{API_URL}/trainingsession/{session_key}",
        headers=HEADERS)
    
    stats = r.json()['trainingSession']['statuses']
    
    training_failed = False
    
    for stat in stats:
        if stat['statusType'] == "TrainingFailed":
            training_failed = True

    if(training_failed):
        print('Training did not succeed')

    return training_failed

def check_training_succeeded(session_key: int):
    r = requests.get(f"{API_URL}/trainingsession/{session_key}",
        headers=HEADERS)
    
    stats = r.json()['trainingSession']['statuses']
    
    training_succeeded = False
    
    for stat in stats:
        if stat['statusType'] == "TrainingCompleted":            
            training_succeeded = True

    if(training_succeeded):
        print('Training succeeded')

    return training_succeeded   

def train_model(
    session_key: int,
    vector_size: int,
    translation_model: bool = True,
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
    
    training_succeeded = False
    training_failed = False

    while(training_failed == False and training_succeeded == False):
        training_failed = check_training_failed(session_key)
        training_succeeded = check_training_succeeded(session_key)
        time.sleep(60)

def translation_training_example():
    """End to end example for translation and gleu scoring"""

    # UPDATE YOUR DESCRIPTION HERE
    session_key = create_session("my test translation session5")
    
    print(f"Created session: {session_key}")   

    training_files = os.listdir(DATA_FOLDER)

    for file in training_files:            
        print(f"Cleaning file {file} before upload.")

        dataset = Path(f"{DATA_FOLDER}/").resolve().joinpath(file)

        clean_file = clean_dataset(dataset)

        upload_training_files(session_key, clean_file)

    print("Upload Completed, Beginning Training")

    training_successful = train_model(session_key, VECTOR_SIZE, translation_model=True)

    if training_successful == False:
        print("Training did not complete successfully")
        return

    print("Training completed successfully, testing inference")

if __name__ == "__main__":
    translation_training_example()