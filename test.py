import requests
from requests.auth import HTTPBasicAuth

url = ""
username = "sms"
password = "kLUPTcJ_"

payload = {
    "message": "có lửa kìa chạy mau",
    "phoneNumbers": [""],
}

response = requests.post(url, json=payload, auth=HTTPBasicAuth(username, password))

print("Response:", response)