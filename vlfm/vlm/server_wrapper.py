# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import os
import random
import socket
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port, threaded=False)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


# def image_to_str(img_np: np.ndarray) -> str:
#     try:
#         # Debugging: Log image properties before encoding
#         if img_np is None:
#             raise ValueError("Input image is None.")
        
#         if not isinstance(img_np, np.ndarray):
#             raise TypeError(f"Expected input to be a NumPy array, got {type(img_np)} instead.")
        
#         if img_np.size == 0:
#             raise ValueError("Input image array is empty.")
        
#         # print(f"[DEBUG] Image shape: {img_np.shape}, dtype: {img_np.dtype}, size: {img_np.size} elements")

#         # Use .png format for lossless compression (no need for quality parameter)
#         retval, buffer = cv2.imencode(".png", img_np)
        
#         if not retval:
#             raise ValueError("Image encoding failed during cv2.imencode.")

#         # Debugging: Log encoded buffer size
#         # print(f"[DEBUG] Encoded buffer size: {len(buffer)} bytes")
        
#         # Encode the image buffer to base64 string
#         img_str = base64.b64encode(buffer).decode("utf-8")
#         # print(f"[DEBUG] Base64 string length: {len(img_str)} characters")
        
#         return img_str

#     except Exception as e:
#         # Capture detailed error information and re-raise
#         print(f"[ERROR] Exception occurred in image_to_str: {str(e)}")
#         raise

def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    # TODO maybe consider a different format for encoding for performance purposes
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str



def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                raise e
                # exit()
            else:
                print(f"Error: {e}. Retrying in 40 seconds...")
                # time.sleep(1)
                time.sleep(40)

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    pid = os.getpid()

    filename = url.replace("/", "_").replace(":", "_") + f"_p-{pid}.lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v)
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        
        slow_request = 'request_timeout' in kwargs # when calling LLM, we increse the amount of time
        if slow_request:
            request_timeout_raise_exception = 50 # in second 
            timeout = 30
            # print("this is a slow request")
        else:
            timeout = 1
            request_timeout_raise_exception = 20
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                elif resp.status_code == 503:  # Server is busy
                    print("Server is busy, retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                print("failed to call the server", url)
                print("HEADERS", headers)
                print("timeout", timeout)
                if time.time() - start_time > request_timeout_raise_exception:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result
