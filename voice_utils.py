# import os
#
#
# from bark import SAMPLE_RATE, generate_audio, preload_models
# from scipy.io.wavfile import write as write_wav
# # from IPython.display import Audio
#
# os.environ["SUNO_OFFLOAD_CPU"] = "True"
# os.environ["SUNO_USE_SMALL_MODELS"] = "True"
#
# # download and load all models
# preload_models()
#
# # generate audio from text
# text_prompt = """
#      Hello, my name is Suno. And, uh — and I like pizza. [laughs]
#      But I also have other interests such as playing tic tac toe.
# """
# audio_array = generate_audio(text_prompt)
#
# # save audio to disk
# write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
#
# # play text in notebook
# # Audio(audio_array, rate=SAMPLE_RATE)


# import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

# import nltk
