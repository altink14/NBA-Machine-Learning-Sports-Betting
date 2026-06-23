import requests
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.fanduel.com/"
}

def test_url(url):
    print(f"\nFetching {url}...")
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print("Status code:", r.status_code)
        print("Headers:", {k: v for k, v in r.headers.items() if k.lower() in ["content-type", "content-length", "server"]})
        if r.status_code == 200:
            try:
                data = r.json()
                print("Data type:", type(data))
                if isinstance(data, dict):
                    print("Keys:", list(data.keys())[:10])
                elif isinstance(data, list):
                    print("List length:", len(data))
                # Save a sample to file
                with open("scratch/fd_api_success.json", "w") as f:
                    json.dump(data, f, indent=2)
                print("Saved response to scratch/fd_api_success.json")
            except Exception as je:
                print("Not JSON response:", je)
                print("First 200 chars:", r.text[:200])
        else:
            print("First 200 chars:", r.text[:200])
    except Exception as e:
        print("Error:", e)

# Test 1: sbapi nj content-managed-page
test_url("https://sbapi.nj.sportsbook.fanduel.com/api/content-managed-page")

# Test 2: sbapi us content-managed-page
test_url("https://sbapi.us.sportsbook.fanduel.com/api/content-managed-page")
