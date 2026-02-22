# MCQ Generation Prompt

You are given a SPICE netlist and its corresponding functional knowledge graph for an analog circuit. Generate a single multiple-choice question (MCQ) in JSON format that tests understanding of the circuit's structure or parameter-performance relationships.

path to netlist:

path to funcitonal graph:

### Output Format

Return ONLY a JSON object with these fields:
```json
{
  "question": "The question text",
  "options": {
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  },
  "answer": "B",
  "circuit_id": "69",
  "netlist": "M1 (VOUT2 VIN2 VSS VSS) nmos4\n..."
}
```