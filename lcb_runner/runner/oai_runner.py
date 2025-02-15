import os
from time import sleep
from typing import Union

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner
# from lcb_runner.runner.parallel_runner import gpts


class OpenAIRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        if 'o1' in model.model_name:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": args.max_tokens,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        try:
            response = OpenAIRunner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return OpenAIRunner._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]

    # def run_batch(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
    #     responses = gpts(prompts)
    #     code_responses = [self._get_code_from_solution(response) for response in responses]
    #     return [list(item) for item in zip(responses, code_responses)]

    def _get_code_from_solution(self, solution: str) -> Union[str, None]:
        '''
        Assume code is in a Markdown block delimited by ```python and ```.
        Returns string of just the code, or None if not found.
        '''
        # try:
        #     begin_delim = "```python"
        #     end_delim = "```"
        #     begin_idx = solution.index(begin_delim)
        #     end_idx = solution.index(end_delim, begin_idx+len(begin_delim))
        #     return solution[begin_idx + len(begin_delim) : end_idx]
        # except Exception as e:
        #     print('Could not parse code from generated solution — returning entire solution')
        #     print(e)
        #     return solution
        parsed = self._extract_code_delim(solution, '```python', '```')
        if parsed:
            return parsed
        parsed = self._extract_code_delim(solution, '```python3', '```')
        if parsed:
            return parsed
        parsed = self._extract_code_delim(solution, '```Python', '```')
        if parsed:
            return parsed
        parsed = self._extract_code_delim(solution, '```Python3', '```')
        if parsed:
            return parsed
        print('Could not parse code from generated solution — returning entire solution')
        return solution

    def _extract_code_delim(self, solution: str, begin_delim: str, end_delim: str) -> Union[str, None]:
        try:
            begin_idx = solution.index(begin_delim)
            end_idx = solution.index(end_delim, begin_idx+len(begin_delim))
            return solution[begin_idx + len(begin_delim) : end_idx]
        except Exception as e:
            print(e)
            return solution

