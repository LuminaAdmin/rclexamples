# RCL EXAMPLES
Before running for the first time, please run the following command
# Environment Setup Instructions
These scripts were written with Python 3.9.
## ANACONDA / SPYDER USERS
1. Install Anaconda: https://www.anaconda.com/
2. Upgrade Conda Python Version: conda install -c anaconda python=3.9
3. Launch Spyder - Open Anaconda Navigator, Find Spyder in the list of applications and click Launch

## Retrieve your API Token
1. Navigate to https://support.lumina247.com
2. Sign in or Signup by clicking the Profile menu
3. Copy your access token from the support site by navigating to the API ACCESS page. Paste your access token into the API_TOKEN variable at the top of the example script.
4. Install nltk library (pip install nltk)
5. Install requests library (pip install requests)
6. Run the example script.

# EXAMPLE SCRIPTS
## rcl_tools_hotword.py
This script demonstrates RCL use for sensor style training data.  We have provided a simple reinforcment_learning.txt dataset for you to try for yourself.  We have seen best results when all lines equal the same amount of sensor inputs and outputs.  

The final value of your input line should be your anticipated result / action to take given the sensor inputs provided during inference.

## rcl_tools_prediction.py
This script demonstrates RCL use for sentence predction.  We have provided a simple_example.txt dataset for you to try for yourself.  Prediction is valuable for convsersational AI / chatbots and the like.

## rcl_tools_search.py
This script demonstrates RCL use for text search.  We have provided a simple_example.txt dataset for you to try for yourself.

## rcl_tools_translation.py
This script demonstrates RCL use for language translation using a simple tab seperated sentence pair dataset.  We have provided a two simple files for translation and an example test set file under the rcl_dataset_translate_test subdirectory.

## rcl_tools_format_tatoeba.py
This script transforms a Tatoeba dataset file into a simple tsv for use with the rcl_tools_translation.py script.  We have provided an English-Yoruba.tsv to test this script with.

# Cleansing Routines
All example scripts will preprocess / cleanse the data in their respective dataset folder, a cleansed copy of the file will be saved locally and postfixed with _cleaned.  The cleansing routine are examples for you to learn from.  Once you've progressed beyond these few examples, you will want to introduce your own cleansing routines that will fit your needs best.