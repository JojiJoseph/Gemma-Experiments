import os
import time
import numpy as np
os.environ["KERAS_BACKEND"] = "torch"  # Or "torch" or "tensorflow".
import torch
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

def gemma_probs(prompt, max_length=1, top_k=5):
    inputs, input_is_scalar = gemma_lm._normalize_generate_inputs(prompt)
    prompt_preprocessed = [gemma_lm.preprocessor.generate_preprocess(
                x, sequence_length=len(prompt) + max_length
            ) for x in inputs]
    # print("prompt_preprocessed", prompt_preprocessed)
    backbone = gemma_lm.backbone
    x = tokenizer.tokenize(prompt)
    # print(x)
    with torch.no_grad():
        next_word_logits = backbone.predict(prompt_preprocessed)
        print(gemma_lm.layers[-1].output_dim)
        token_embedding = gemma_lm.layers[-1]
        next_word_logits = token_embedding(next_word_logits, reverse=True)
        best_word = next_word_logits[0,len(x)].cpu().numpy().argmax()
    # print(best_word, print(tokenizer.detokenize([best_word])))
    top_k_word = next_word_logits[0,len(x)].cpu().numpy().argsort()[-top_k:][::-1]
    print(list(tokenizer.detokenize([word_id]).numpy().decode("utf-8") for word_id in top_k_word))
    # backbone_cls = gemma_lm.
    # next_word_logits = gemma_lm.predict(next_word_logits)
    # exit()
    # print("next_word_logits", next_word_logits.shape)
    return prompt + str(tokenizer.detokenize([top_k_word[np.random.randint(0, len(top_k_word))]]).numpy().decode("utf-8"))
generated = gemma_lm.generate(prompt, max_length=256)
print(generated)

pre_prompt = prompt
for i in range(100):
    print("Step ", i)
    prompt = gemma_probs(prompt, max_length=1, top_k=2)
    if prompt == pre_prompt:
        break
    pre_prompt = prompt
    print(prompt)