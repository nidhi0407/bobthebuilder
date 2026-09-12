"""
End-to-End Automated Test Script for BobTheBuilder Safety Alert Server
"""

import sys
import time
import requests

# Reconfigure stdout for UTF-8 compatibility on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 60)
    print("[TEST] Running BobTheBuilder Safety Server Test Suite")
    print("=" * 60)

    # 1. Test GET /status
    print("\n[1] Testing GET /status...")
    res = requests.get(f"{BASE_URL}/status")
    assert res.status_code == 200, f"Failed status check: {res.status_code}"
    status_data = res.json()
    print("    Initial Status:", status_data)

    # 2. Test GET /worker and GET /forklift
    print("\n[2] Testing GET /worker and GET /forklift...")
    res_w = requests.get(f"{BASE_URL}/worker")
    assert res_w.status_code == 200 and "<html" in res_w.text.lower(), "Worker page failed to serve"
    print(f"    Worker Page served OK ({len(res_w.content)} bytes)")

    res_f = requests.get(f"{BASE_URL}/forklift")
    assert res_f.status_code == 200 and "<html" in res_f.text.lower(), "Forklift page failed to serve"
    print(f"    Forklift Page served OK ({len(res_f.content)} bytes)")

    # 3. Test POST /trigger
    print("\n[3] Testing POST /trigger with custom alert messages...")
    payload = {
        "worker_message": "Immediate danger! Forklift entering blind corner on your right!",
        "forklift_message": "Emergency stop! Worker detected in vehicle corridor 4 meters ahead!"
    }
    res_trigger = requests.post(f"{BASE_URL}/trigger", json=payload)
    assert res_trigger.status_code == 200, f"Trigger failed: {res_trigger.status_code}"
    trigger_data = res_trigger.json()
    print(f"    Trigger Response: {trigger_data['status']}")
    print(f"    Active State: {trigger_data['state']['active']}")
    print(f"    Worker Msg:   {trigger_data['state']['worker_message']}")
    print(f"    Forklift Msg: {trigger_data['state']['forklift_message']}")
    print(f"    Timestamp:    {trigger_data['state']['timestamp']}")
    assert trigger_data['state']['active'] is True, "State should be active"
    assert trigger_data['state']['worker_message'] == payload['worker_message']
    assert trigger_data['state']['forklift_message'] == payload['forklift_message']

    # 4. Test GET /audio/worker.mp3 and /audio/forklift.mp3
    print("\n[4] Testing Audio File Endpoints...")
    res_waudio = requests.get(f"{BASE_URL}/audio/worker.mp3?t={trigger_data['state']['timestamp']}")
    assert res_waudio.status_code == 200, f"Worker audio failed: {res_waudio.status_code}"
    assert len(res_waudio.content) > 1000, f"Worker audio file suspiciously small: {len(res_waudio.content)} bytes"
    assert res_waudio.headers.get("Cache-Control"), "Missing Cache-Control header"
    print(f"    Worker Audio: OK ({len(res_waudio.content)} bytes, Content-Type: {res_waudio.headers.get('Content-Type')})")

    res_faudio = requests.get(f"{BASE_URL}/audio/forklift.mp3?t={trigger_data['state']['timestamp']}")
    assert res_faudio.status_code == 200, f"Forklift audio failed: {res_faudio.status_code}"
    assert len(res_faudio.content) > 1000, f"Forklift audio file suspiciously small: {len(res_faudio.content)} bytes"
    print(f"    Forklift Audio: OK ({len(res_faudio.content)} bytes, Content-Type: {res_faudio.headers.get('Content-Type')})")

    # 5. Test POST /clear
    print("\n[5] Testing POST /clear...")
    res_clear = requests.post(f"{BASE_URL}/clear")
    assert res_clear.status_code == 200, f"Clear failed: {res_clear.status_code}"
    clear_data = res_clear.json()
    assert clear_data['state']['active'] is False, "State should not be active after clear"
    print(f"    Clear Response: {clear_data['status']}")
    print(f"    Active State:   {clear_data['state']['active']}")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
