#!/usr/bin/env python3
"""Send sample JSON inference requests to the board-side TCP endpoint.

Usage: python3 test_client.py <board-host> [port]
Example: python3 test_client.py 127.0.0.1 8765
"""

import json
import socket
import sys

def send_request(host: str, port: int, payload: dict) -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(360)
        s.connect((host, port))
        s.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        s.shutdown(socket.SHUT_WR)   # Signal end of request data.
        raw = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            raw += chunk
    return json.loads(raw.decode("utf-8"))

def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

    # Use one domain payload and one generic English payload.
    test_cases = [
        {
            "prompt": "A traffic camera detected 12 vehicles, 3 pedestrians, and an average speed of 45 km/h. Analyze the intersection and provide advice.",
            "tokens": 64,
            "temperature": 600,
            "top_k": 40,
        },
        {
            "prompt": "Hello, what is FPGA?",
            "tokens": 16,
            "temperature": 0,
            "top_k": 1,
        },
    ]

    for i, payload in enumerate(test_cases):
        print(f"\n=== Test {i+1} ===")
        print(f"Request: {json.dumps(payload, ensure_ascii=False)[:120]}")
        try:
            response = send_request(host, port, payload)
            print(f"Status: {response.get('status')}")
            print(f"Text: {response.get('text', '')}")
            print(f"Token count: {response.get('token_count', 0)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
