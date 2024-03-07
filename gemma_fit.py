import os
import time
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
import keras
import keras_nlp
from load_secret import load_secret
import numpy as np

LOAD_WEIGHTS = True
TRAIN = False
def main():

    load_secret()

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.backbone.enable_lora(rank=4)
    gemma_lm.preprocessor.sequence_length = 512
    gemma_lm.summary()
    optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
    gemma_tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_2b_en")
    sampler = keras_nlp.samplers.TopKSampler(k=1, seed=time.time_ns())
    if LOAD_WEIGHTS:
        gemma_lm.load_weights("gemma_lm.weights.h5")
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        sampler=sampler,
    )

    # Exclude layernorm and bias terms from decay.
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])


    data = []
    data_x = []
    for i in range(1000):
        num = int(np.random.random() * 10000)
        rev = str(num)[::-1]
        query = f"Reverse {num}\n Answer: {rev}\n"
        data.append(query)
        data_x.append(f"Reverse {num}\n")
    if TRAIN:
        gemma_lm.fit(data[:900], epochs=2, batch_size=5)
        gemma_lm.save_weights("gemma_lm.weights.h5")
    response = gemma_lm.generate(data_x[900:905], max_length=25)
    print(response)
    response = gemma_lm.generate(["Reverse 15.38\n", "Reverse asd\n", "Reverse 0x1B\n", "Reverse 1.23\n", "Reverse b110 \n"], max_length=25)
    print(response)

if __name__ == "__main__":
    main()