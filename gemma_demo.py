import os
import time
os.environ["KERAS_BACKEND"] = "torch"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
import keras
import keras_nlp
def load_secret():
    import json
    import os

    secret = json.load(open("kaggle.json"))

    os.environ["KAGGLE_USERNAME"] = secret["username"]
    os.environ["KAGGLE_KEY"] = secret["key"]
load_secret()

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_instruct_2b_en")

gemma_lm.summary()

# print("gemma_lm", gemma_lm)
sampler = keras_nlp.samplers.TopKSampler(k=1, seed=2)
gemma_lm.compile(sampler=sampler)

prompt = """
How are you?
"""

generated = gemma_lm.generate(prompt, max_length=256)
print(generated)