import requests

# URL of the audio file
url = "https://gotranscript.com/audios/practice/transcribing_2.mp3?v=546f7b323209cc223603e4b81b53fc22"
output_filename = "transcribing_2.mp3"

try:
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode and save the content
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded audio file as '{output_filename}'")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")