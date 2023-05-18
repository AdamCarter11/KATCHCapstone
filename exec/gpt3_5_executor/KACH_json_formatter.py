import json
from collections import defaultdict

# Load the JSON data
with open('response.json', 'r') as f:
    data = json.load(f)

# Initialize a dictionary to store the timestamp of each tag occurrence
tag_timestamps = defaultdict(list)
transcript_segments = []

# Note: we assume that 'tags' is a list of dictionaries, where each dictionary contains a 'tag' key and a 'position' key.
# 'position' is also a list of dictionaries, and each of those contains a 'wordIndex' and a 'timestamp'.
for tag_info in data['tags']:
    tag = tag_info['tag']
    for pos_info in tag_info['position']:
        word_index = pos_info['wordIndex']
        timestamp = pos_info['timestamp']
        tag_timestamps[tag].append(timestamp)

        transcript_segment = data['allText'].split()[word_index-10:word_index+10]  # Get +/- 10 words around the tag
        transcript_segments.append({
            'timestamp': timestamp,
            'transcript': ' '.join(transcript_segment)
        })

# Prepare the final dictionary
video_metadata = {
    "video_1": {
        "video_tags": list(tag_timestamps.keys()),
        "transcript_segments": transcript_segments,
        "tag_timestamps": dict(tag_timestamps)
    }
}

# Save the final dictionary into a new JSON file
with open('formatted_response.json', 'w') as f:
    json.dump(video_metadata, f, indent=2)
