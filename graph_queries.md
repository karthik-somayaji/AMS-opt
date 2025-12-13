# G_p Query:
Performance metrics: gain, CMRR, UGF. 

Identify relationships among these performance metrics for the given circuit netlist. Output entries of the form (Metric A, Relationship, Metric B), where Relationship can only be of type ‘trade-off’, ‘directly-proportional’, ‘ambiguous’. 

Netlist:

# G_ps Query:
Given the performance-graph (P-graph) and circuit netlist, identify the circuit substructures that influence performance metrics in the P-graph. Output tuples of the form (Substructure, 'influences', Metric), to connect circuit sub-strucutres with performance metrics, if the substructure significantly influences a performance metric.

P-graph:

Netlist:

# G_psx Query
Given the P-Graph and PS-Graph, identify relationships between design parameters X, circuit substructures S, and performance metrics P. Output entries of the form (Parameter, Relationship, Substructure) where Relationship is ‘belongs-to’ if a parameter belongs to a sub-structure. 
Next output entries of the form (Parameter, Relationship, Metric) where Relationship can be ‘directly-proportional’ if increase in parameter value increases metric value and vice versa or ‘inversely-proportional’ if increase in parameter value decreases metric value and vice versa.

P-graph:

PS-graph:

Netlist:


##################################
New Prompt:

Given a circuit netlist and a set of performance metrics, do the following and output everything together as plain tuple lists, without any explanations:

1. **Infer P-graph (performance–performance relations).**
   Output entries of the form:
   `(Metric_A, Relationship, Metric_B)`
   where `Relationship ∈ { 'trade-off', 'directly-proportional', 'ambiguous' }`.

2. **Infer PS-graph (substructure–performance relations).**
   First, identify meaningful circuit substructures (e.g., “M0-M1 differential pair”, “M2-M3 active load current mirror”, “R0-R1 load resistors”).
   Then output entries of the form:
   `(Substructure, 'influences', Metric)`
   only if that substructure significantly affects the metric.

3. **Infer PSX-graph (parameter–substructure–performance relations).**
   Treat device sizes, bias currents, resistances, etc. as design parameters (e.g., `W_M0, L_M0, IB1, R0, R1, VB1`).
   Output:

   * `(Parameter, 'belongs-to', Substructure)` if the parameter structurally belongs to that substructure.
   * `(Parameter, Relationship, Metric)` where `Relationship ∈ { 'directly-proportional', 'inversely-proportional' }`, indicating how increasing the parameter value affects the metric.

**Important formatting constraints:**

* Use exactly the tuple formats above.
* Use only the allowed relationship labels.
* Do **not** include any explanations, text, or headings—only the tuples.

Given the above instructions, construct the above tuples as entries of a knowledge graph. Specifically, output a json format where all nodes (nodes in P, PS and PSX graphs) are listed under the field 'nodes'.
For example:
"nodes": [
    {"id": "Active load",              "type": "substructure"},
    {"id": "CMRR",                     "type": "performance"},
    {"id": "Differential pair",        "type": "substructure"}, etc]
    
 All tuples in the graphs should be listed under a 'links' field as a list of dictionaries.
 For example: "links": [
    {"source": "Gain", "target": "CMRR", "relation": "ambiguous"}, etc]


Netlist:

