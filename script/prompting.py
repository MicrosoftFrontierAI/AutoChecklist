import json
from pathlib import Path
import numpy as np
from typing import List, Optional, Any, Callable, Hashable
import time


DEFAULT_SLEEP = 120 #seconds

# caching stuff
CACHE_FILE = Path(__file__).parent.parent / "output" / "cache.json"
CACHE_FILE.parent.mkdir(exist_ok=True)
CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


# =============================================================================
# ====================  PLUG YOUR LLM CLIENT IN HERE  =========================
# =============================================================================
# This is the only place that talks to an LLM. The original internal client has
# been removed for public release. Replace the body of `call_llm` below with a
# call to whatever model / endpoint you use (OpenAI, Azure OpenAI, Anthropic,
# a local server, etc.). Keep the input and output contract identical so the
# rest of the pipeline keeps working unchanged.
#
# INPUT
#   model        : str   -- model / deployment name. If the name contains the
#                           substring "chat", the pipeline routes to the chat
#                           endpoint; otherwise it uses the completion endpoint.
#   request_data : dict  -- request payload already assembled by `get_response`.
#                           For "completion": has keys
#                               prompt, max_tokens, temperature, top_p, n,
#                               stream, stop, logprobs
#                           For "chat": has keys
#                               messages (list of {"role", "content"}),
#                               max_tokens, temperature, top_p, n, stream, stop
#   endpoint     : str   -- either "completion" or "chat".
#
# EXPECTED RETURN (OpenAI-style response dict)
#   For "completion":
#       {"choices": [
#           {"text": "...",
#            "logprobs": {"token_logprobs": [<float>, ...]}},
#           ...
#       ]}
#   For "chat":
#       {"choices": [
#           {"message": {"role": "assistant", "content": "..."}},
#           ...
#       ]}
#   Return None (or raise) on unrecoverable failure.
# =============================================================================


class RateLimitError(Exception):
    """Raise this from `call_llm` to trigger the built-in back-off / retry."""
    pass


def call_llm(model: str, request_data: dict, endpoint: str):
    """Send a single request to your LLM and return an OpenAI-style response.

    See the contract documented in the banner above. Raise `RateLimitError`
    to have `_wrapped_llm_client_send_request` sleep and retry automatically.
    """
    raise NotImplementedError(
        "No LLM client configured. Implement `call_llm` in "
        "script/prompting.py with your own model/endpoint. "
        "See the contract documented above this function."
    )


def _wrapped_llm_client_send_request(request_data, model:str = "gpt-4o-chat-completions", endpoint:str="completion", num_tries:int =8):
    # retry with back-off on rate limits
    for calls_left in reversed(range(0, num_tries)):
        try:
            start_time = time.time()
            response = call_llm(model, request_data, endpoint)
            end_time = time.time()
            time_spent = end_time-start_time
            return response |{"response_time":time_spent}
        except (RateLimitError) as err:
            print(err)
            seconds = DEFAULT_SLEEP
            print(
                f"Sleeping {seconds}s due to rate limit (tries left={calls_left})")
            time.sleep(seconds)
    return None


def get_response(history: List, prompt: str, model: Optional[str]=None, max_tokens: Optional[int]=250, temperature: Optional[float]=0.0, n: Optional[int]=1, stop: Optional[List[str]]=None, logprobs: Optional[int]=1, return_raw_response: bool= False) -> tuple[List[str], float]:
    """Get completion for a prompt."""
    stop = stop or "\n"
    request_data = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "n": n,
        "stream": False,
        "stop": stop,
    }
    
    endpoint = "completion"
    if model and "chat" in model:
        endpoint = "chat"

    if endpoint == "completion":
        request_data['prompt']=prompt
        request_data['logprobs']=logprobs
    elif endpoint == "chat":
        input = []
        f=1
        if history==[]:
            f=0
            history.append(prompt.split("<user>")[0].strip())
            for segments in prompt.split("<user>")[1:]:
                if "<assistant>" not in segments:
                    history.append(segments.strip())
                else:
                    history.append(segments.split("<assistant>")[0].strip())
                    history.append(segments.split("<assistant>")[1].strip())
          
        input.append({"role": "system", "content": history[0]} )
        for i in range(1,len(history)):
            if i%2:
                input.append({"role": "user", "content": history[i]})
            else:
                input.append({"role": "assistant", "content": history[i]})
        if f==1:
            history.append(prompt.split("<user>")[1].strip())
            input.append({"role": "user", "content": history[-1]})
        request_data["messages"]= input

    key = _key(request_data, model)
    if key not in CACHE:
        raw_response = _wrapped_llm_client_send_request(request_data, model, endpoint)
        # only add to cache if actually worked
        if raw_response is not None and isinstance(raw_response, dict) and "choices" in raw_response:
            CACHE[key] = {
                "request": request_data,
                "response": raw_response
            }
            CACHE_FILE.write_text(json.dumps(CACHE))
        else:
            return []
    response = CACHE[key]["response"]
    if response is None:
        return []
    if endpoint == "completion":
        unique_choices = get_unique_in_order(response["choices"], key=lambda x: x["text"])
        unique_choices = [
            entry for entry in unique_choices if len(entry["text"].strip()) > 0
        ]
        unique_choices = sorted(
            unique_choices,
            key=lambda x: np.mean(x["logprobs"]["token_logprobs"]),
            reverse=True,
        )
    elif endpoint == "chat":
        unique_choices = get_unique_in_order(response["choices"], key=lambda x: x["message"]['content'])
        unique_choices = [
            entry for entry in unique_choices if len(entry["message"]['content'].strip()) > 0
        ]
   
    if "response_time" in response:
        response_time= response["response_time"]
    else:
        ### loading from cache with no "response_time" key in it. return 0
        response_time = None

    if endpoint == "completion":
        resp = [e["text"] for e in unique_choices] if not return_raw_response else unique_choices
    elif endpoint == "chat":
        resp = [e["message"]["content"] for e in unique_choices] if not return_raw_response else unique_choices
    history.append(resp[0])
    return resp, response_time, history


def _key(dict: dict, model):
    dict = dict.copy()
    dict["model"] = model
    return json.dumps(dict, sort_keys=True)


def get_unique_in_order(
    lst: List[Any], key: Optional[Callable[[Any], Hashable]] = None
):
    uniq = []
    seen = set()
    for elem in lst:
        k = key(elem) if key is not None else elem
        if k not in seen:
            seen.add(k)
            uniq.append(elem)
    return uniq
