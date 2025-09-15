*------------------------------------------------------------
* Repaired / runnable HSPICE netlist (gpt-5 base)
*------------------------------------------------------------

.option post=2 nomod gmin=1e-12 rshunt=1e12
.temp 27
.param VDD=1.8

* Rails: node 1 = VDD, node 0 = GND
V1 1 0 DC 'VDD'

* Load / sense resistor
R1 1 2 20k

*-------------------------
* PMOS array (sources/bulks @ VDD)
* Mx: D  G  S  B  MODEL ...
*-------------------------
M1  2 10 1 1 PMOS W=20u  L=1u
M2  2  9 1 1 PMOS W=10u  L=1u
M3  2  6 1 1 PMOS W=20u  L=1u
M4  2  7 1 1 PMOS W=10u  L=1u
M5  2  8 1 1 PMOS W=5u   L=1u
M6  2 11 1 1 PMOS W=80u  L=1u
M7  2 12 1 1 PMOS W=40u  L=1u
M8  2 13 1 1 PMOS W=20u  L=1u
M9  2 14 1 1 PMOS W=80u  L=1u

*-------------------------
* NMOS bank (sources/bulks @ GND)
*-------------------------
M10 5  4 0 0 NMOS W=128u L=1u
M11 5  0 0 0 NMOS W=1u   L=1u
M12 6  5 0 0 NMOS W=32u  L=1u
M13 8  5 0 0 NMOS W=32u  L=1u
M14 9  5 0 0 NMOS W=32u  L=1u
M15 10 5 0 0 NMOS W=64u  L=1u
M16 11 5 0 0 NMOS W=16u  L=1u
M17 12 5 0 0 NMOS W=8u   L=1u
M18 13 4 0 0 NMOS W=128u L=1u

*-------------------------
* DC current sources (reinterpreting node "3" as GND)
* Iname N+ N- value  (flows N+ -> N-)
*-------------------------
I2 0 8 DC 10uA
I3 0 4 DC 10uA
I4 2 5 DC 5uA
I5 2 9 DC 2.5uA

* Replace floating I1 with a small load sink at node 2
I1 2 0 DC 10uA

*-------------------------
* Simple placeholder MOS models (replace with your PDK)
*-------------------------
.model NMOS nmos level=1 VTO=0.6  KP=200u  LAMBDA=0.02 GAMMA=0.5 PHI=0.6
.model PMOS pmos level=1 VTO=-0.6 KP=100u  LAMBDA=0.02 GAMMA=0.5 PHI=0.6

*-------------------------
* Analyses / outputs
*-------------------------
.op
.print dc V(1) V(2) I(V1)
.probe DC ALL

.end