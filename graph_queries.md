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
