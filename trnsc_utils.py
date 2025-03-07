import whisper
from pathlib import Path
import os
import time

os.environ["PATH"] += r";C:\path\to\ffmpeg\bin"

folder_name = Path(os.path.dirname(os.path.abspath(__file__)))
file_path = folder_name / "handling.m4a"

model = whisper.load_model("medium")

start_time = time.time()
result = model.transcribe(str(file_path),
                          temperature=0.1,
                          beam_size=15,
                          word_timestamps=True,
                          language="ru")
end_time = time.time()

with open("trnsc_result_medium2.txt", "w") as f_write:
    f_write.write(result["text"])
    f_write.write(f"\ntime: {round(end_time - start_time, 1)} sec")

for i in result["text"].split('.'):
    print(i)
