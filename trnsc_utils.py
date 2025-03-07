import whisper
from pathlib import Path
import os


os.environ["PATH"] += r";C:\path\to\ffmpeg\bin"

folder_name = Path(os.path.dirname(os.path.abspath(__file__)))
file_path = folder_name / "handling.m4a"

model = whisper.load_model("large")

#result = model.transcribe("audio.mp3", temperature=0.1)
# result = model.transcribe("audio.mp3", beam_size=10) default beam_size=5
# result = model.transcribe("audio.mp3", word_timestamps=True)
# result = model.transcribe("audio.mp3", language="ru")  # Russian


result = model.transcribe(str(file_path),
                           temperature=0.1,
                           beam_size=15,
                           word_timestamps=True,
                           language="ru")

with open("trnsc_result_large1.txt", "w") as f_write:
    f_write.write(result["text"])

for i in result["text"].split('.'):
    print(i)