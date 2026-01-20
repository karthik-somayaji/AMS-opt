import os
import random
import math
import time
import openai
import asyncio
import numpy as np
import pandas as pd
from aiohttp import ClientSession


from .utils import RateLimiter

#openai.api_key = '[REDACTED_KEY]'
my_key = "[REDACTED_KEY]"
my_key_confirmation = "[REDACTED_KEY]"
openai.api_key = "[REDACTED_KEY]"

class GPT(object):
    def __init__(self,
                 model="3.5",
                 seed=114514,
                 max_token=500,
                 temperature=0.5,
                 rate_limiter=None,
                 n_gen=5,
                 debug_mode=False
                 ):
        self.n_gen = n_gen
        if model == "4":
            self.model = "gpt-4-turbo"
        else:
            self.model = "gpt-3.5-turbo"

        self.seed = seed
        self.max_token = max_token
        self.temperature = temperature

        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=60000, time_frame=60)
            #self.rate_limiter = RateLimiter(max_tokens=40000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter

        self.debug_mode = debug_mode

    def request(self, prompt):
        message = []
        message.append({"role": "system", "content": "You are an AI assistant that helps people find information."})
        message.append({"role": "user", "content": prompt})

        MAX_RETRIES = 3

        response_raw = None
        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                self.rate_limiter.add_request(request_text=prompt, current_time=start_time)
                response_raw = openai.ChatCompletion.create(
                    model=self.model,  # Use the model identifier for ChatGPT-3.5
                    messages=message,
                    seed=self.seed,
                    max_tokens=self.max_token,
                    temperature=self.temperature,
                    n=self.n_gen
                )
                self.rate_limiter.add_request(request_token_count=response_raw['usage']['total_tokens'],
                                              current_time=start_time)
                break
            except Exception as e:
                print(f'[AF] RETRYING LLM REQUEST {retry + 1}/{MAX_RETRIES}...')
                print(response_raw)
                print(e)

        if response_raw is None:
            return None

        response = []
        for i in range(self.n_gen):
            response.append(response_raw['choices'][i]['message']['content'])

        if self.debug_mode:
            for r in response:
                print(r)

        return response





if __name__ == "__main__":
    import openai

    # Make sure to replace 'your-api-key-here' with your actual OpenAI API key if not using environment variables
    openai.api_key = '[REDACTED_KEY]'

    # mseg = "You are an experienced analog circuit designer who is asked to optimize the sizing for a circuit. The circuit is a two stage amplifier with 8 transistors (M1 to M8). M1 and M2 are current mirror. M3 to M6 are the first stage of the amplifier. M7 and M8 are the second stage of the amplifier. You need to adjust the sizes all transistors to achieve optimal performance.\
    #         The adjustable parameters are width and length of all transistors: ['w1', 'l1'', 'w2', 'l2', ..., 'w8', 'l8']. Each parameter is normalized to be within range [0., 1.].\
    #         The performance of the circuit is defined as the gain of the circuit.\
    #         Below are some collected parameter setting and the corresponding performance:\
    #         params: [0.65610115, 0.35778869, 0.52839249, 0.77677977, 0.4958911, 0.77201432, 0.66606936, 0.38786399, 0.67393735, 0.45725441, 0.48214926, 0.3202441 , 0.13345133, 0.17212094, 0.99388058, 0.13647008], performance: 1.0\
    #         params: [0.42475653, 0.76520546, 0.622838, 0.14970467, 0.9056879, 0.14020604, 0.05920767, 0.55384032, 0.77470616, 0.98518153, 0.41966512, 0.38235059, 0.81321267, 0.08155391, 0.96936949, 0.90329737], performance: 2.0\
    #         recommend 2 new parameter settings that will achieve a target of 3.0.\
    #         Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest possible precision.\
    #         Your response must only contain the parameters in the format ## params ##"

    prompt = """
    assume you are an analog circuit design expert. Here is a Hspice file (amp2.sp) of a two-stage differential amplifier, whose sizes are defined in param.inc file. Please [1] provide good parameters in param.inc to get high gain performance, and [2] explain your design choice step by step.
[amp2.sp]
* Sim
.inc ./param.inc
** Differential amplifier for process variation analysis
.INCLUDE "/data/llmbo/tech/hspice.include"
.inc /data/llmbo/tech/nfet_perfect.inc
.inc /data/llmbo/tech/pfet_perfect.inc

** Circuit Netlist
.GLOBAL vdd!

.TEMP 25
.OPTION
+    ARTIST=2
+    INGOLD=2
+    MEASOUT=1
+    PARHIER=LOCAL
+    PSF=2
+	 OPFILE=1

cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! vb 5e-6

xp1 node1 node1 vdd! vdd! pfet l=l1 w=w1
xp2 out1 node1 vdd! vdd!  pfet l=l2 w=w2
xn1 node1 vip node2 0     nfet l=l3 w=w3
xn2 out1 vin node2 0      nfet l=l4 w=w4
xnb vb vb 0 0             nfet l=l5 w=w5
xnc node2 vb 0 0          nfet l=l6 w=w6
xpo out out1 vdd! vdd!    pfet l=l7 w=w7
xno out vb 0 0            nfet l=l8 w=w8
cc out outm 'c1'
rz outm out1 'r1'
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the model identifier for ChatGPT-3.5
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    print(response['choices'][0]['message']['content'])
