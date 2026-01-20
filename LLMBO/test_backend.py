if __name__ == "__main__":
    import openai
    import re

    # Make sure to replace 'your-api-key-here' with your actual OpenAI API key if not using environment variables
    openai.api_key = ""

    # mseg = "You are an experienced analog circuit designer who is asked to optimize the sizing for a circuit. The circuit is a two stage amplifier with 8 transistors (M1 to M8). M1 and M2 are current mirror. M3 to M6 are the first stage of the amplifier. M7 and M8 are the second stage of the amplifier. You need to adjust the sizes all transistors to achieve optimal performance.\
    #         The adjustable parameters are width and length of all transistors: ['w1', 'l1'', 'w2', 'l2', ..., 'w8', 'l8']. Each parameter is normalized to be within range [0., 1.].\
    #         The performance of the circuit is defined as the gain of the circuit.\
    #         Below are some collected parameter setting and the corresponding performance:\
    #         params: [0.65610115, 0.35778869, 0.52839249, 0.77677977, 0.4958911, 0.77201432, 0.66606936, 0.38786399, 0.67393735, 0.45725441, 0.48214926, 0.3202441 , 0.13345133, 0.17212094, 0.99388058, 0.13647008], performance: 1.0\
    #         params: [0.42475653, 0.76520546, 0.622838, 0.14970467, 0.9056879, 0.14020604, 0.05920767, 0.55384032, 0.77470616, 0.98518153, 0.41966512, 0.38235059, 0.81321267, 0.08155391, 0.96936949, 0.90329737], performance: 2.0\
    #         recommend 2 new parameter settings that will achieve a target of 3.0.\
    #         Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest possible precision.\
    #         Your response must only contain the parameters in the format ## params ##"
    """
    Based on the netlist and demonstrations, summarize how you can adjust the sizing of parameters to increase the gain, increase the cmrr, increase the gbw and increase the pm. 
    """
    prompt = """
You are an analog circuit design expert. **Circuit Netlist** gives an HSPICE netlist of a two-stage differential amplifier. 
You need to optimize the size of these parameters: [l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, l6, w6, l7, w7, l8, w8, r1, c1] to achieve the desired performance on the following metrics:  
gain: >60 dB, Common-Mode Rejection Ratio (cmrr): >80 dB, Gain Bandwidth Product (gbw): >1 MHz, Phase Margin (pm): >=45 degree.
**Examples** gives demonstration of existing parameter settings and the simulated metrics from HSPICE. 
Based on the netlist and demonstrations, explain how you can adjust the sizing of parameters to achieve the desired specifications. 
Then propose a different set of parameters in the format of params in **Examples**. 
You must not add any comments beyond the recommendation.

**Circuit Netlist**
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

**Examples**
params: w1=100u, l1=1u, w2=100u, l2=1u, w3=10u, l3=1u, w4=10u, l4=1u, w5=5u, l5=1u, w6=5u, l6=1u, w7=100u, l7=1u, w8=10u, l8=1u, r1=1k, c1=1p
metrics: gain=15.574 gbw=1119296.542 cmrr=84.496 pm=40.620

params: w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=2p r1=1k
metrics: gain=49.570 cmrr=97.229 gbw=2755006.863 pm=11.056

params: w1=200u l1=1u w2=200u l2=1u w3=20u l3=1u w4=20u l4=1u w5=15u l5=1u w6=15u l6=1u w7=200u l7=1u w8=20u l8=1u c1=2p r1=1k
metrics: gain=50.422 cmrr=90.162 gbw=2533316.381 pm=6.311

params: w1=250u l1=1u w2=250u l2=1u w3=25u l3=1u w4=25u l4=1u w5=20u l5=1u w6=20u l6=1u w7=250u l7=1u w8=25u l8=1u c1=2p r1=1k
metrics: gain=50.702 cmrr=87.538 gbw=2366425.544 pm=3.224

params: w1=300u l1=1u w2=300u l2=1u w3=30u l3=1u w4=30u l4=1u w5=20u l5=1u w6=20u l6=1u w7=200u l7=1u w8=20u l8=1u c1=2p r1=1k
metrics: gain=49.424 cmrr=88.936 gbw=2208463.466 pm=-0.877

params: w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=1.5k
metrics: gain=49.688 cmrr=100.177 gbw=3160489.981 pm=8.753

params: w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=2k
metrics: gain=49.688 cmrr=100.177 gbw=3161074.529 pm=9.313

params: w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=800
metrics: gain=49.688 cmrr=100.177 gbw=3160222.244 pm=7.963

params: w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=4p r1=800
metrics: gain=49.570 cmrr=97.229 gbw=1971151.431 pm=25.441

params: w1=42u l1=0.65u w2=37u l2=0.55u w3=28u l3=0.45u w4=24u l4=0.35u w5=20u l5=0.3u w6=17u l6=0.25u w7=28u l7=0.55u w8=24u l8=0.4u r1=7.5k c1=70p

    """


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the model identifier for ChatGPT-3.5
        #model="gpt-4-turbo",  # Use the model identifier for ChatGPT-3.5
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=1000,
        temperature=0.5,
        n=5
    )

    r = []
    for i in range(5):
        r.append(response['choices'][i]['message']['content'])

    # for ri in r:
    #     print("***\n")
    #     print(ri)
    #     print("***\n")

    # print(response['choices'][0]['message']['content'])

    def parse_llm_response(responses):
        params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        params_list = ["w1", "l1", "w2", "l2", "w3", "l3", "w4", "l4", "w5", "l5", "w6", "l6", "w7", "l7", "w8", "l8", "c1", "r1"]

        def extract_params(response):
            found_values = {}
            # Create a regular expression pattern for the units, allowing only specified units;
            # Escape units to avoid regex issues;
            units_pattern = f"[{''.join(re.escape(unit) for unit in params_units)}]?"

            for param in params_list:
                # Regex to find the parameter followed by a number (including decimal numbers) and optionally a unit, allowing spaces;
                match = re.search(rf"{re.escape(param)}\s*=\s*(\d+(\.\d+)?{units_pattern})\b", response)
                if match:
                    # Store the found value using the parameter as the key;
                    found_values[param] = match.group(1)
                else:
                    # If any parameter is not found or its value is invalid, return False;
                    return False
            def convert_value_to_float(found_values):
                # convert the values to float; eg. 1.0u to 1e-6, 1.0k to 1e3, etc.
                dic_items = list(found_values.items())
                for param, value in dic_items:
                    # Find the last character of the value;
                    unit = value[-1]
                    # If the last character is a unit, convert the value to a float;
                    if unit in params_units:
                        # Convert the value to a float;
                        value = float(value[:-1])
                        # Convert the value to the correct unit;
                        if unit == "f":
                            value *= 1e-15
                        elif unit == "p":
                            value *= 1e-12
                        elif unit == "n":
                            value *= 1e-9
                        elif unit == "u":
                            value *= 1e-6
                        elif unit == "k":
                            value *= 1e3
                        elif unit == "M":
                            value *= 1e6
                        elif unit == "G":
                            value *= 1e9
                        # Update the value in the dictionary;
                        found_values[f"{param}_float"] = value
                    else:
                        # If the last character is not a unit, convert the value to a float;
                        found_values[f"{param}_float"] = float(value)
            formatted_output = ".param " + " ".join(f"{param}={value}" for param, value in found_values.items())
            convert_value_to_float(found_values)
            print(found_values)
            return formatted_output

        params_parsed = []
        for response in responses:
            response_parsed = extract_params(response)
            if response_parsed:
                params_parsed.append(response_parsed)

        return params_parsed

    params = parse_llm_response(r)

    print(r[0])