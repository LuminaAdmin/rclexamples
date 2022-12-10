from typing import Any
import requests
import json

# This example script demonstrates how to start an inference host
# PLEASE SET THE FOLLOWING VARIABLES
# 1. API_TOKEN (retrieve from the support portal @ https://support.lumina247.com)
# 2. SET SESSION_KEY this is the session_key for your trained model

API_TOKEN = "bearer eyJraWQiOiJpZzA2bDFkQjhlQkUxanAxN0tkYTMxUlVNVEM4QVpVdXVFVGd4aHQ1ZWQ0PSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiJkZjFmYWNhNi1mYmNmLTQzNGItOTE2Ny1hYzFmNjM0NDU4M2EiLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAudXMtZWFzdC0xLmFtYXpvbmF3cy5jb21cL3VzLWVhc3QtMV9zYkNlc1k5eWMiLCJ2ZXJzaW9uIjoyLCJjbGllbnRfaWQiOiIzcDM2c2xsZTc1ZjYzcXJpMHRhczZldG9zNSIsIm9yaWdpbl9qdGkiOiI4ZWNhZjFiYy00MTFiLTQ2MjEtOGNiYi0xOWE5NTc5ZjNjZTIiLCJldmVudF9pZCI6IjRjMTY1MWI0LTE2N2YtNDY5ZS05NzhhLTJkNzYzZmIyYTJlOCIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUiLCJhdXRoX3RpbWUiOjE2NzA2OTQ3NDksImV4cCI6MTY3MDc4MTE1MSwiaWF0IjoxNjcwNjk0NzUxLCJqdGkiOiJhYjNhMTVhOS00YWFjLTQ3ODMtYmIwZC1lYzkwZmI0OTg3Y2EiLCJ1c2VybmFtZSI6ImRmMWZhY2E2LWZiY2YtNDM0Yi05MTY3LWFjMWY2MzQ0NTgzYSJ9.DHT7AVM6CTCc9PAO_cDf16VjgDiYssvfco7JC6z6To9yCcKuIpm7wb7aVH27NKGE5UE6k5PpEX60aTVWDsGq_8OXq7QIFVyXWDjJH_NdDtUNh6ZoJUgQZqlUd1OUO0qf35oP3ur9BvCqlOCf4vrdRctzwskSiM7W8jNRFCcRcXfWi2nxpHi50tc1hpspZ3LNTuFYWgm3R--OAUHhp5akAhqVE7klumnyvD0cqcgtrMsqiFh5vzZusq2hG38CqQKgkZynenQosN8C8p-z7tQQu3JZBQsjPI60eA6EiG5w4H6RUo-sQfyOoAsXuxwGOBzTvNtMbS5PSyCrap3t1ZXT7Q"

# SET YOUR TRAINING SESSION KEY HERE
SESSION_KEY = "107879761"

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
        f"/start"        
    )

    r = requests.post(endpoint, data=json.dumps(SESSION_KEY), headers=HEADERS)
    
    if r.status_code != 200:
        raise Exception(f"Error calling inference: {r.json()}")
    else:
        print('INFERENCE HOST STARTING')

if __name__ == "__main__":
    stop_host()