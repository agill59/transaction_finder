import json

def time_to_seconds(time_str):
    """Converts a MM:SS string into total integer seconds."""
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def clean_transactions(input_file="src/transactions.json", output_file="src/transactions_clean.json", threshold_seconds=5):
    # 1. Load the raw data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return

    if not data:
        print("The JSON file is empty.")
        return

    # 2. Sort data alphabetically by video, then chronologically by time
    data.sort(key=lambda x: (x['video_name'], time_to_seconds(x['timestamp'])))

    condensed_data = []
    
    # 3. Setup the tracking variables for the first item
    current_event = data[0]
    last_seen_time = time_to_seconds(current_event['timestamp'])

    # 4. Loop through and condense duplicates
    for i in range(1, len(data)):
        item = data[i]
        item_time = time_to_seconds(item['timestamp'])

        # Check if it's the same video AND within the 3-second rolling window
        if item['video_name'] == current_event['video_name'] and (item_time - last_seen_time) <= threshold_seconds:
            # It's part of the same transaction event. 
            # We update the 'last_seen_time' to push the 3-second window forward.
            last_seen_time = item_time
        else:
            # The gap was larger than 3 seconds, or it's a new video.
            # Save the previous event and start tracking the new one.
            condensed_data.append(current_event)
            current_event = item
            last_seen_time = item_time

    # Don't forget to append the very last event in the list!
    condensed_data.append(current_event)

    # 5. Save the cleaned data to a new file
    with open(output_file, 'w') as f:
        json.dump(condensed_data, f, indent=4)

    print(f"✅ Cleanup complete!")
    print(f"Original detections: {len(data)}")
    print(f"Condensed events: {len(condensed_data)}")
    print(f"Saved to {output_file}")

# Run the function
if __name__ == "__main__":
    clean_transactions()