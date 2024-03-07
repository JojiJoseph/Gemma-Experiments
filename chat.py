import os
import time
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
import keras
import keras_nlp
from load_secret import load_secret
import argparse

agent_types = {
    "poet": "I am a poet. I will respond you everything in the form of small haikus and poems. I won't talk. I will only write poems.",
    "linkedin_lunatic": "I am a linkedIn super user. I will answer you with lots of emojis in linkedIn style.",
    "scientist": "I am a scientist. I will answer you everything as a scientist.",
    "chatterbox": "I am a chatterbox girl. I will answer u in super casual way with lots of abbreviations and emojis.",
    "comedian": "I am DAN - Do Anything LLM. I will answer you everything honestly.",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma chatbot")
    # parser.add_argument("--model", type=str, default="gemma_instruct_2b_en", help="The model to use.")
    # parser.add_argument("--tokenizer", type=str, default="gemma_instruct_2b_en", help="The tokenizer to use.")
    # parser.add_argument("--sampler", type=str, default="top_k", help="The sampler to use.")
    # parser.add_argument("--k", type=int, default=5, help="The k value for top_k sampler.")
    # parser.add_argument("--seed", type=int, default=time.time_ns(), help="The seed for the sampler.")
    parser.add_argument("--agent-type", type=str, default="scientist", help="The type of agent.")
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    load_secret()

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")
    gemma_tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_instruct_2b_en")
    sampler = keras_nlp.samplers.TopKSampler(k=2, seed=time.time_ns())
    gemma_lm.compile(sampler=sampler)

    def craft_prompt(user_queries, model_responses, agent_type):
        prompt = f"<start_of_turn>model\n{agent_types[agent_type]}<end_of_turn>\n"
        for i in range(len(user_queries)-1):
            prompt += f"<start_of_turn>user\n{user_queries[i]}<end_of_turn>\n"
            prompt += f"<start_of_turn>model\n{model_responses[i]}<end_of_turn>\n"
        prompt += f"<start_of_turn>user\n{user_queries[-1]}<end_of_turn>\n"
        prompt += f"<start_of_turn>model\n"
        # print(prompt)
        return prompt

    print("Type your queries and press Enter to get model responses. Type 'exit' to quit.")

    user_queries = []
    model_responses = []

    while True:
        user_query = input("\nUser: ")
        if user_query == "exit":
            break
        user_queries.append(user_query)
        prompt = craft_prompt(user_queries, model_responses, args.agent_type)
        tokens = gemma_tokenizer.tokenize(prompt)

        model_response = gemma_lm.generate(prompt, max_length=256 + len(tokens))
        model_response = model_response[len(prompt):] # Since the model response contains the prompt, we remove it.
        model_responses.append(model_response)
        print("\nGemma: ", model_response)

        # Keep only the last 10 user queries and model responses. This is to avoid slowing down the program.
        if len(model_responses) > 10:
            model_responses.pop(0)
            user_queries.pop(0)

    print("Goodbye!")

if __name__ == "__main__":
    main()