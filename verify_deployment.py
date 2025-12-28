import requests
import sys
import time
from datetime import datetime

SERVICES = [
    {
        "name": "Weather API",
        "url": "http://localhost:8000/health",
        "type": "api"
    },
    {
        "name": "Airflow Webserver",
        "url": "http://localhost:8080/health",
        "type": "web"
    },
    {
        "name": "MLflow Tracking",
        "url": "http://localhost:5050",
        "type": "web"
    },
    {
        "name": "Jupyter Lab",
        "url": "http://localhost:8889",
        "type": "web"
    }
]

def check_service(service):
    try:
        response = requests.get(service["url"], timeout=5)
        if response.status_code == 200:
            return True, f"✅ {service['name']} is UP ({service['url']})"
        else:
            return False, f"⚠️ {service['name']} return code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ {service['name']} is DOWN (Connection refused)"
    except Exception as e:
        return False, f"❌ {service['name']} Error: {str(e)}"

def verify_deployment():
    print(f"🚀 Starting System Verification at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    all_up = True
    for service in SERVICES:
        status, message = check_service(service)
        print(message)
        if not status:
            all_up = False
            
    print("=" * 50)
    if all_up:
        print("🎉 ALL SYSTEMS GO! Deployment is healthy.")
        # Try a prediction
        try:
            print("\n🧪 Testing Prediction Endpoint...")
            payload = {
                "features": {
                    "temperature_2m_max": 25.0,
                    "temperature_2m_min": 15.0,
                    "precipitation_sum": 0.0,
                    "windspeed_10m_max": 10.0
                }
            }
            resp = requests.post("http://localhost:8000/predict", json=payload)
            if resp.status_code == 200:
                print(f"✅ Prediction Successful: {resp.json()}")
            else:
                print(f"❌ Prediction Failed: {resp.text}")
        except Exception as e:
            print(f"❌ Prediction Error: {e}")
    else:
        print("⚠️ Some services are not ready yet. Please wait a few moments and try again.")

if __name__ == "__main__":
    verify_deployment()
