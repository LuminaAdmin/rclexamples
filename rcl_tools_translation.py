from collections import Counter
from typing import Any
import requests
from pathlib import Path
import random
import json
import time
from enum import Enum
from nltk.util import everygrams
from requests import Response
import re
import os

#THIS SCRIPT MUST BE PROVIDED A SIMPLE TAB SEPERATED SENTENCE PAIR DATASET TO FUNCTION PROPERLY
#EACH SENTENCE MUST BE TERMINATED WITH PUNCUTATION
#EACH LINE MUST BE TERMINATED WITH A CARRIAGE RETURN
#IF USING TATOEBA DATASETS FOR TRAINING:
# 1. PLEASE FOLLOW CLEANSING ROUTINE IN rcl_tools_format_tatoeba 
# 2. PROVIDE THIS SCRIPT THE OUTPUT DATASET CREATED BY STEP 1

# Please set API_TOKEN to what is displayed from https://support.lumina247.com, please make sure the token is in the format "bearer <token>"
# "PASTE TOKEN HERE"
API_TOKEN = "PASTE TOKEN HERE"

# Default chunk size for uploads to 50MB
FILE_UPLOAD_CHUNK_SIZE = 50000000

# The vector size to use for training, best results tend to be between 3-5, you may need to adjust for best results
VECTOR_SIZE = 5

# You may change this line to point to a custom dataset for training
DATA_FOLDER = "rcl_dataset_translate"

TEST_PATH = "rcl_dataset_translate_test/dan_test.txt"

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

def sentence_gleu(references, hypothesis, min_len=1, max_len=4):
    """
    Calculates the sentence level GLEU (Google-BLEU) score described in

        Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi,
        Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey,
        Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens,
        George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith,
        Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes,
        Jeffrey Dean. (2016) Google’s Neural Machine Translation System:
        Bridging the Gap between Human and Machine Translation.
        eprint arXiv:1609.08144. https://arxiv.org/pdf/1609.08144v2.pdf
        Retrieved on 27 Oct 2016.

    From Wu et al. (2016):
        "The BLEU score has some undesirable properties when used for single
         sentences, as it was designed to be a corpus measure. We therefore
         use a slightly different score for our RL experiments which we call
         the 'GLEU score'. For the GLEU score, we record all sub-sequences of
         1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
         compute a recall, which is the ratio of the number of matching n-grams
         to the number of total n-grams in the target (ground truth) sequence,
         and a precision, which is the ratio of the number of matching n-grams
         to the number of total n-grams in the generated output sequence. Then
         GLEU score is simply the minimum of recall and precision. This GLEU
         score's range is always between 0 (no matches) and 1 (all match) and
         it is symmetrical when switching output and target. According to
         our experiments, GLEU score correlates quite well with the BLEU
         metric on a corpus level but does not have its drawbacks for our per
         sentence reward objective."

    Note: The initial implementation only allowed a single reference, but now
          a list of references is required (which is consistent with
          bleu_score.sentence_bleu()).

    The infamous "the the the ... " example

        >>> ref = 'the cat is on the mat'.split()
        >>> hyp = 'the the the the the the the'.split()
        >>> sentence_gleu([ref], hyp)  # doctest: +ELLIPSIS
        0.0909...

    An example to evaluate normal machine translation outputs

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands').split()
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party').split()
        >>> hyp2 = str('It is to insure the troops forever hearing the activity '
        ...            'guidebook that party direct').split()
        >>> sentence_gleu([ref1], hyp1) # doctest: +ELLIPSIS
        0.4393...
        >>> sentence_gleu([ref1], hyp2) # doctest: +ELLIPSIS
        0.1206...

    :param references: a list of reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :return: the sentence level GLEU score.
    :rtype: float
    """
    return corpus_gleu([references], [hypothesis], min_len=min_len, max_len=max_len)

def corpus_gleu(list_of_references, hypotheses, min_len=1, max_len=4):
    """
    Calculate a single corpus-level GLEU score (aka. system-level GLEU) for all
    the hypotheses and their respective references.

    Instead of averaging the sentence level GLEU scores (i.e. macro-average
    precision), Wu et al. (2016) sum up the matching tokens and the max of
    hypothesis and reference tokens for each sentence, then compute using the
    aggregate values.

    From Mike Schuster (via email):
        "For the corpus, we just add up the two statistics n_match and
         n_all = max(n_all_output, n_all_target) for all sentences, then
         calculate gleu_score = n_match / n_all, so it is not just a mean of
         the sentence gleu scores (in our case, longer sentences count more,
         which I think makes sense as they are more difficult to translate)."

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> corpus_gleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
    0.5673...

    The example below show that corpus_gleu() is different from averaging
    sentence_gleu() for hypotheses

    >>> score1 = sentence_gleu([ref1a], hyp1)
    >>> score2 = sentence_gleu([ref2a], hyp2)
    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS
    0.6144...

    :param list_of_references: a list of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :return: The corpus-level GLEU score.
    :rtype: float
    """
    # sanity check
    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"

    # sum matches and max-token-lengths over all sentences
    corpus_n_match = 0
    corpus_n_all = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(everygrams(hypothesis, min_len, max_len))
        tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(everygrams(reference, min_len, max_len))
            # True positives + False negatives.
            tpfn = sum(ref_ngrams.values())

            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())  # True positives.

            # While GLEU is defined as the minimum of precision and
            # recall, we can reduce the number of division operations by one by
            # instead finding the maximum of the denominators for the precision
            # and recall formulae, since the numerators are the same:
            #     precision = tp / tpfp
            #     recall = tp / tpfn
            #     gleu_score = min(precision, recall) == tp / max(tpfp, tpfn)
            n_all = max(tpfp, tpfn)

            if n_all > 0:
                hyp_counts.append((tp, n_all))

        # use the reference yielding the highest score
        if hyp_counts:
            n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all

    # corner case: empty corpus or empty references---don't divide by zero!
    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all

    return gleu_score

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

def inference_host_ready_check(
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
    
    if "unrecognized model" in str(r.content):
        return inference_host_ready_check(session_key, priority, detail)

    return r

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
    
    current_status = ""
    inference_ready_check = 0

    while(current_status != "trainingcompleted" and current_status != "trainingfailed"):
        info = get_session_info(session_key)
        current_status = info["trainingsession"]["statuses"][-1]["statusTypeName"].lower()
        print(f"Current Status: {current_status}")
        time.sleep(60)
    
    print (f"Health check complete, training status is {current_status}")

    while(inference_ready_check != 200):
        inference_ready_check = inference_host_ready_check(session_key, InferencePriorityType.index, InferenceDetailType.search).status_code   
    
    if current_status == "trainingcompleted":
        return True
    else: 
        return False

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

def evaluate_translation_model(
    session_key: int, path: Path, sample_size=100, iterations=5
):
    for i in range(iterations):
        start = time.monotonic()

        sample = sample_translation_set(
            path=path, sample_size=sample_size, vector_size=VECTOR_SIZE
        )

        data = [
            (
                *s,
                p := inference(
                    session_key,
                    s[0],
                    InferencePriorityType.index,
                    InferenceDetailType.translate_line,
                ),
                p == s[1],
                sentence_gleu([s[1].split()], p.split()),
            )
            for s in sample
        ]
        # data = [(source, target, prediction, accuracy, gleu)]

        accuracy = round(sum(map(lambda x: int(x[3]), data)) / len(sample), 2)
        gleu_avg = round(sum(map(lambda x: x[4], data)) / len(sample), 2)

        end = time.monotonic()
        elapsed = round(end - start, 2)

        results = (
            f"Iteration:   ({i+1}/{iterations})\n"
            f"Sample size: {len(sample)}\n"
            f"Accuracy:    {accuracy}\n"
            f"GLEU:        {gleu_avg}\n"
            f"Time:        {elapsed}s"
        )

        print(results)

        header = "Source\tTarget\tTranslation\tAccuracy\tGLEU"
        lines = [
            results,
            "\n===\n",
            header,
            *["\t".join(map(str, d)) for d in data if not d[3]],
        ]
        results_file = Path(f"{session_key} TRANSLATION_RESULTS {i+1}.txt")
        results_file.write_text("\n".join(lines), encoding="utf8")
        print(f"Results written to {str(results_file.absolute())}")

def translation_example():
    """End to end example for translation and gleu scoring"""

    # UPDATE YOUR DESCRIPTION HERE
    session_key = create_session("my test translation session5")
    
    print(f"Created session: {session_key}")   

    test_set = Path(TEST_PATH)

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
   
    sample = "It is bad manners to point at people."

    print(f"Example input: {sample}")

    result = inference(
        session_key,
        sample,
        InferencePriorityType.index,
        InferenceDetailType.translate_line,
    )

    print("Example output: " + result)

    print("Evaluating GLEU Scores")

    evaluate_translation_model(session_key, test_set)

if __name__ == "__main__":
    translation_example()