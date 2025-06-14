# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
import os
import json
from time import sleep
from datetime import datetime

with open("cookies.txt", "r") as file:
    cookies = file.read().strip()

headers = {
    "cookie": cookies
}

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14, 23, 59, 59)

OUTPUT_FOLDER = "discourse"
os.makedirs(OUTPUT_FOLDER, exist_ok = True)

page=0

while True:
    print(f"Fetching page {page}...")
    url = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json?page={page}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch page {page}")
        break
    
    topics = response.json().get("topic_list", {}).get("topics", [])
    if not topics:
        print(f"No more topics found on page {page}.")
        break


    for topic in topics:
        try:
            topic_date = datetime.strptime(topic["created_at"], DATE_FORMAT)
        except Exception:
            print(f"  - Skipping topic with bad date format: {topic.get('id')}")
            continue

        if START_DATE <= topic_date <= END_DATE:
            topic_id = topic["id"]
            topic_title = topic["title"]
            topic_slug = topic["slug"]
            filename_safe_title = "".join(c if c.isalnum() or c in '-_' else '_' for c in topic_title)[:50]
            filename = f"{topic_id}_{filename_safe_title}.json"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            print(f"Fetching topic: {topic_id} ({topic_title[:40]})...")

            topic_url = f"{BASE_URL}/t/{topic_id}.json"
            topic_res = requests.get(topic_url, headers=headers)

            if topic_res.status_code == 200:
                topic_data = topic_res.json()
                thread = {
                    "id" : topic_id,
                    "title": topic_title,
                    "created_at": topic["created_at"],
                    "posts" : topic_data.get("post_stream", {}).get("posts", [])
                }

                with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(thread, f, indent=2, ensure_ascii=False)
            else:
                print(f"Failed to fetch topic {topic_id}")
            
            sleep(0.5)

    page += 1
    sleep(1)

print(f"✅ Done!")
