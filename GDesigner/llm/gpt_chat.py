import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from openai import OpenAI, AsyncOpenAI

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7']
BASE_URL = 'https://api.shubiaobiao.cn/v1/'

load_dotenv()
MINE_BASE_URL = "https://api.shubiaobiao.cn/v1/"
MINE_API_KEYS = "sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7"


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    # request_url = MINE_BASE_URL
    # authorization_key = MINE_API_KEYS
    # headers = {
    #     'Content-Type': 'application/json',
    #     'authorization': authorization_key
    # }
    # data = {
    #     "name": model,
    #     "inputs": {
    #         "stream": False,
    #         "msg": repr(msg),
    #     }
    # }
    # async with aiohttp.ClientSession() as session:
    #     async with session.post(request_url, headers=headers ,json=data) as response:
    #         response_data = await response.json()
    #         prompt = "".join([item['content'] for item in msg])
    #         cost_count(prompt,response_data['data'],model)
    #         return response_data['data']
    client = AsyncOpenAI(base_url = MINE_BASE_URL, api_key = MINE_API_KEYS, max_retries=0)
    chat_completion = await client.chat.completions.create(messages = msg, model = model, temperature=1, seed=123)
    response = chat_completion.choices[0].message.content
    prompt_tokens = chat_completion.usage.prompt_tokens
    completion_tokens = chat_completion.usage.completion_tokens
    prompt = "".join([item['content'] for item in msg])
    price, prompt_len, completion_len = cost_count(prompt,response,model,prompt_tokens,completion_tokens)
    return response, price, prompt_len, completion_len

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        # if max_tokens is None:
        #     max_tokens = self.DEFAULT_MAX_TOKENS
        # if temperature is None:
        #     temperature = self.DEFAULT_TEMPERATURE
        # if num_comps is None:
        #     num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass