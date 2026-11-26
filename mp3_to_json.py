import whisper
import json
import os

# Load Whisper model for English audio transcription
model = whisper.load_model("large-v2")

# Get audio files from the audios folder 
audios = os.listdir("audios")

for audio in audios: 
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)
        result = model.transcribe(audio = f"audios/{audio}", # Transcribe audio and generate timestamped segments
                              language="en",
                              task="transcribe",
                              word_timestamps=False )

        # Store each segment with lecture metdata and timestamps
        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})

        # Save the transcript and segments as a JSON file
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        
        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)