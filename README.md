# RCL TRANSLATION EXAMPLES
These example scripts are intended to run in order
1_rcl_training_translation.py - Train a new model.
2_rcl_inference_translation.py - Start your inference host and test an inference example.
3_rcl_training_translation_gleu.py - Test Gleu and Accuracy scores for a trained translation model.
4_rcl_start_inference_host.py - Start an inference host.
5_rcl_stop_inference_host.py - Stop an inference host.
6_rcl_terminate_inference_host.py - Permanently terminate an inference host.
rcl_example_format_tatoeba.py - Example script on how to prepare a Tatoeba sentence pair file for training

Each script has special instructions in the head of the script, please follow these instructions closely and input the required variable values
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
1_rcl_tools_training_translation.py - This script trains your translation model.
2_rcl_tools_inference_translation - This script starts your inference host and runs an example inference on your trained model.
3_rcl_tools_training_translation_gleu - This script performs gleu scores on your trained model.
4_start_inference_host.py - This script starts your inference host.
5_stop_inference_host.py - This script stops your inference host.  This is helpful for when you may not need your model hosted for a period of time.
6_terminate_inference_host.py - This script PERMANENTLY terminates your inference host.  Executing this script is an unreversable action and your trained model will no longer be available for use.