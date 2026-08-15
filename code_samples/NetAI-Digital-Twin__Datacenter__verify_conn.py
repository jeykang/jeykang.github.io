import requests
import json
import time
import logging
from datetime import datetime, timedelta
import pytz

"""
v0: Made by Anv (~24-08-28)
"""


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def json_read(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
urls = json_read('my_private_setting.json')

# Elasticsearch and Kibana URLs
kibana_url = urls["kibana_url"]
es_url = urls["es_url"]

# Kibana headers for API requests
headers = {
    "kbn-xsrf": "true",
    "Content-Type": "application/json",
}

# List of object IDs to fetch data for
obj_ids = [20, 21, 22, 23, 24, 25, 191, 192]

def get_latest_index_name():
    # Endpoint to retrieve all index patterns
    index_patterns_url = f"{kibana_url}/api/saved_objects/_find?type=index-pattern"
    response = requests.get(index_patterns_url, headers=headers, verify=False)
    if response.status_code == 200:
        index_patterns = response.json()
        # Filter patterns that match the 'perfhist-fms' prefix
        relevant_patterns = [pattern for pattern in index_patterns['saved_objects'] if pattern['attributes']['title'].startswith('perfhist-fms')]
        # Sort by updated_at to get the latest one
        latest_pattern = max(relevant_patterns, key=lambda x: x['updated_at'])
        return latest_pattern['attributes']['title']
    else:
        raise ValueError(f"Failed to retrieve index patterns: {response.status_code} - {response.text}")

def fetch_data_for_objid(obj_id, index_name):
    try:
        # Define the query to search for documents by objId and timestamp
        seoul_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(seoul_tz)
        one_minute_ago = current_time - timedelta(minutes=1)
        query = {
            "_source": ["TEMPERATURE", "HUMIDITY", "@timestamp"],  # Fetch temperature, humidity, and timestamp fields
            "query": {
                "bool": {
                    "must": [
                        {"term": {"objId": obj_id}},
                        {"range": {"@timestamp": {"gt": one_minute_ago.isoformat()}}}
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "asc"}}]  # Sort by timestamp in ascending order
        }

        es_search_url = f"{es_url}/{index_name}/_search"
        es_response = requests.post(es_search_url, headers={"Content-Type": "application/json"}, data=json.dumps(query), verify=False)

        if es_response.status_code == 200:
            es_data = es_response.json()
            hits = es_data['hits']['hits']
            if hits:
                for hit in hits:
                    source = hit['_source']
                    temperature = source.get('TEMPERATURE')
                    humidity = source.get('HUMIDITY')
                    timestamp = source.get('@timestamp')
                    if temperature is not None and humidity is not None and timestamp is not None:
                        # Convert the timestamp from UTC to KST
                        utc_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                        kst_time = utc_time.replace(tzinfo=pytz.utc).astimezone(seoul_tz)
                        formatted_timestamp = kst_time.strftime("%B %dth %Y, %H:%M:%S.%f")[:-3]
                        logger.info(f"Document ID: {hit['_id']} for objId: {obj_id}")
                        logger.info(f"Temperature: {temperature} °C")
                        logger.info(f"Humidity: {humidity} %")
                        logger.info(f"Timestamp: {formatted_timestamp}")
                        logger.info("-" * 40)
                    else:
                        logger.info(f"Skipping document ID: {hit['_id']} for objId: {obj_id} due to missing temperature, humidity, or timestamp.")
            else:
                logger.info(f"No new document found with objId {obj_id} in index '{index_name}'.")
        else:
            logger.error(f"Failed to fetch data from Elasticsearch: {es_response.status_code} - {es_response.text}")
    except Exception as e:
        logger.error(f"Error connecting to Kibana API or Elasticsearch: {e}")

def main():
    while True:
        # Retrieve the latest index pattern name
        index_name = get_latest_index_name()
        logger.info(f"Using index: {index_name}")

        for obj_id in obj_ids:
            fetch_data_for_objid(obj_id, index_name)
        time.sleep(59)  # Updated sleep interval to 59 seconds

if __name__ == "__main__":
    main()
