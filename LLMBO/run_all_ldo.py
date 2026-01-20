from os import system

models = [
         
        "DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "GPT-4o",
        "llama3-70B-instruct",
        "Mistral-Small-24B-Instruct-2501",
        "Qwen2.5-32B-Instruct",
        
        ]

system(f"python llmbo_ldo.py --history 0 --refined 0 --model GPT-4o  --related_ckts \"Two_Stage_Differential_Amplifier\"")
for model in models:
    system(f"python llmbo_ldo.py --history 1 --refined 0 --model {model}  --related_ckts \"Two_Stage_Differential_Amplifier\"")
for model in models:
    system(f"python llmbo_ldo.py --history 1 --refined 1 --model {model}  --related_ckts \"Two_Stage_Differential_Amplifier\"")


# system(f"python llmbo.py --history 0 --refined 0 --model GPT-4o  --related_ckts \"Two_Stage_Differential_Amplifier\" \"Hysteresis_Comparator\" ")
# for model in models:
#     system(f"python llmbo.py --history 1 --refined 0 --model {model}  --related_ckts \"Two_Stage_Differential_Amplifier\" \"Hysteresis_Comparator\" ")
# for model in models:
#     system(f"python llmbo.py --history 1 --refined 1 --model {model}  --related_ckts \"Two_Stage_Differential_Amplifier\" \"Hysteresis_Comparator\" ")