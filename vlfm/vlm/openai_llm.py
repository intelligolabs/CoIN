import os
from colorama import Fore
from colorama import init as init_colorama
from openai import OpenAI

init_colorama(autoreset=True)
from retrying import retry
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


class OpenAILLMClient:
    def __init__(self, llm_client_params) -> None:
        print(Fore.YELLOW + f"[INFO] Initializing OpenAI LLM")

        all_env_vars = os.environ

        self.api_keys = [value for key, value in all_env_vars.items() if key.startswith("COIN_LLM_CLIENT_KEY")]
        llm_client_params["api_key"] = self.api_keys[0] if self.api_keys else None
        self.model = llm_client_params.get("model", "gpt-4o")
        del llm_client_params["model"]

        self.client = OpenAI(**llm_client_params)

    @retry(
        retry_on_exception=(
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            Exception,
        ),
        stop_max_attempt_number=1,
        wait_fixed=6000,
    )
    def ask(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                top_p=1,
                max_tokens=3000,
                seed=42,
            )
            return completion.choices[0].message.content

        except RateLimitError as e:
            print(Fore.RED + "[ERROR] Rate Limit Error")
            print(Fore.RED + f"[ERROR] {e}")
            raise Exception("retry")


if __name__ == "__main__":
    ## Test with python vlfm/vlm/openai_llm.py

    # you can also use Groq for testing, otherwise it will use OpenAI
    ### make sure to set the environment variable LLM_CLIENT_KEY (inside .env.llm_client_key) to either your OpenAI API key or groq API key
    # If using Groq, register here for a free api (https://groq.com/)
    test_with_groq = True

    if test_with_groq:
        llm_client_params = {
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
        }
    else:
        llm_client_params = {
            "model": "gpt-4o",
        }

    llm_client = OpenAILLMClient(llm_client_params)

    prompt = "What is the capital of France?"
    response = llm_client.ask(prompt)
    print(Fore.GREEN + f"Response: {response}")
