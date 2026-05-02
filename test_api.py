import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "op1": 0,
    "op2": 0,
    "op3": 0,
    "sensor_1": 518,
    "sensor_2": 642,
    "sensor_3": 1590,
    "sensor_4": 1400,
    "sensor_5": 14,
    "sensor_6": 21,
    "sensor_7": 554,
    "sensor_8": 2388,
    "sensor_9": 9046,
    "sensor_10": 1,
    "sensor_11": 47,
    "sensor_12": 521,
    "sensor_13": 2388,
    "sensor_14": 8138,
    "sensor_15": 8,
    "sensor_16": 0,
    "sensor_17": 392,
    "sensor_18": 2388,
    "sensor_19": 100,
    "sensor_20": 39,
    "sensor_21": 23
}

response = requests.post(url, json=data)
print(response.json())