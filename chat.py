import os
import time
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
import keras
import keras_nlp
from load_secret import load_secret

def main():

    load_secret()

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")
    gemma_tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_instruct_2b_en")
    sampler = keras_nlp.samplers.TopKSampler(k=5, seed=time.time_ns())
    gemma_lm.compile(sampler=sampler)

    def craft_prompt(user_queries, model_responses):
        prompt = ""
        for i in range(len(user_queries)-1):
            prompt += f"<start_of_turn>user\n{user_queries[i]}<end_of_turn>\n"
            prompt += f"<start_of_turn>model\n{model_responses[i]}<end_of_turn>\n"
        prompt += f"<start_of_turn>user\n{user_queries[-1]}<end_of_turn>\n"
        prompt += f"<start_of_turn>model\n"
        return prompt

    print("Type your queries and press Enter to get model responses. Type 'exit' to quit.")

    user_queries = []
    model_responses = []

    while True:
        user_query = input("User: ")
        if user_query == "exit":
            break
        user_queries.append(user_query)
        prompt = craft_prompt(user_queries, model_responses)
        tokens = gemma_tokenizer.tokenize(prompt)

        model_response = gemma_lm.generate(prompt, max_length=256 + len(tokens))
        model_response = model_response[len(prompt):] # Since the model response contains the prompt, we remove it.
        model_responses.append(model_response)
        print("Gemma: ", model_response)

        # Keep only the last 10 user queries and model responses. This is to avoid slowing down the program.
        if len(model_responses) > 10:
            model_responses.pop(0)
            user_queries.pop(0)

    print("Goodbye!")

if __name__ == "__main__":
    main()