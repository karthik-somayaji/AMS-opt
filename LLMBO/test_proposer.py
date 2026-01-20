import os
import random
import math
import time
import openai
import asyncio
import numpy as np

from backend.llm import gpt
from core import proposer

if __name__ == "__main__":

    context = "You are an experienced analog circuit designer who is asked to optimize the sizing for a circuit. The circuit is a two stage amplifier with 8 transistors (M1 to M8).\
                        M1 and M2 are current mirror. M3 to M6 are the first stage of the amplifier. M7 and M8 are the second stage of the amplifier.\
                        You need to adjust the sizes all transistors to achieve optimal performance."

    b = gpt.GPT()
    p = proposer.Proposer(task_context=context, backend=b)



    data_collected = {
        "params": np.random.rand(5, 16),
        "metrics": np.random.randn(5, 4),
        "targets": np.random.randn(5),
        "aux_info": [[''] for _ in range(5)]
    }

    data_proposed = p.propose_params(data_collected)