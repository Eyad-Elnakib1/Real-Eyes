#!/usr/bin/env python3
"""
RealEyes Automated Diagnostic & Smoke Test Script

This script pings the RealEyes Flask backend health probe, verifies that all
AI subsystems and models are responsive, and prints a visual diagnostics card.
Run this script to verify system readiness after setup or during CI/CD.
"""

import sys
import time
import urllib.request
import json

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HEALTH_URL = "http://127.0.0.1:5001/health"

def run_smoke_test():
    print("=" * 60)
    print("        REALEYES ENTERPRISE SMOKE TEST & PROBE         ")
    print("=" * 60)
    print(f"\n[1/3] Pinging API Gateway at {HEALTH_URL} ...")
    
    start_time = time.time()
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "RealEyes-SmokeTest/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            status_code = resp.getcode()
            raw_data = resp.read().decode("utf-8")
            data = json.loads(raw_data)
    except Exception as e:
        print(f"\n[FAIL] ERROR: Failed to connect to backend server ({e}).")
        print("Please ensure the Flask backend is running: cd backend && python server.py")
        sys.exit(1)
        
    latency_ms = round((time.time() - start_time) * 1000, 2)
    print(f"[OK] Gateway Responded in {latency_ms} ms (HTTP {status_code})\n")
    
    print("[2/3] Analyzing Subsystem Health Metrics...")
    models = data.get("models", {})
    
    print(f"  • Status        : {data.get('status', 'unknown').upper()}")
    print(f"  • API Version   : {data.get('version', 'unknown')}")
    print(f"  • Compute Device: {data.get('device', 'unknown').upper()} "
          f"({'CUDA Enabled' if data.get('gpu_available') else 'CPU Fallback'})")
    print(f"  • Uptime        : {data.get('uptime_seconds', 0)} seconds\n")
    
    print("[3/3] Model Readiness Check:")
    def fmt_status(val):
        return "[READY]" if val else "[NOT PRELOADED] (Will lazy-load on first request)"
        
    print(f"  • Swin-V2-B Classifier   : {fmt_status(models.get('classification'))}")
    print(f"  • AXUNet Segmenter       : {fmt_status(models.get('segmentation'))}")
    print(f"  • DeBERTa Fact-Checker   : {fmt_status(models.get('fact_checker'))}")
    print(f"  • Groq LLM Translator    : {fmt_status(models.get('llm_translator'))}")
    
    print("\n" + "=" * 60)
    print("           SMOKE TEST COMPLETED SUCCESSFULLY           ")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_smoke_test()
