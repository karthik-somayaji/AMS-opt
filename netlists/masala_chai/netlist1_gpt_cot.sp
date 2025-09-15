* ==============================
* Executable HSPICE Netlist (gpt-5 CoT prompt)
* ==============================

.TITLE Binary-weighted PMOS array with NMOS gate tree (DAC-like output)

* -------- Parameters & Supply --------
.PARAM VDD=1.8
VDD  vdd  0  DC {VDD}

* -------- Load to create an analog output node --------
* Node "vout" corresponds to original node 2
R1   vdd  vout  20k
Cout vout 0 100f

* -------- Simple MOS models (no PDK needed) --------
.MODEL NMOS NMOS LEVEL=1 VTO=0.5  KP=200e-6  GAMMA=0.4 PHI=0.65 LAMBDA=0.02
.MODEL PMOS PMOS LEVEL=1 VTO=-0.5 KP=80e-6   GAMMA=0.4 PHI=0.65 LAMBDA=0.02

* -------- PMOS array: drains -> vout, sources/bulks -> vdd --------
* Mname D     G    S    B    Model  W     L
M1   vout n10  vdd  vdd  PMOS  W=20u  L=1u
M2   vout n9   vdd  vdd  PMOS  W=10u  L=1u
M3   vout n6   vdd  vdd  PMOS  W=20u  L=1u
M4   vout n7   vdd  vdd  PMOS  W=10u  L=1u
M5   vout n8   vdd  vdd  PMOS  W=5u   L=1u
M6   vout n11  vdd  vdd  PMOS  W=80u  L=1u
M7   vout n12  vdd  vdd  PMOS  W=40u  L=1u
M8   vout n13  vdd  vdd  PMOS  W=20u  L=1u
M9   vout n14  vdd  vdd  PMOS  W=80u  L=1u

* -------- NMOS gate-pull tree (sources/bulks -> 0) --------
* Mname D    G   S  B   Model   W      L
M10  n5   n4  0  0   NMOS  W=128u L=1u
M11  n5   0   0  0   NMOS  W=1u   L=1u      * gate=0 -> off; left in place
M12  n6   n5  0  0   NMOS  W=32u  L=1u
M13  n8   n5  0  0   NMOS  W=32u  L=1u
M14  n9   n5  0  0   NMOS  W=32u  L=1u
M15  n10  n5  0  0   NMOS  W=64u  L=1u
M16  n11  n5  0  0   NMOS  W=16u  L=1u
M17  n12  n5  0  0   NMOS  W=8u   L=1u
M18  n13  n4  0  0   NMOS  W=128u L=1u

* -------- Pull-ups on PMOS gate nodes (define DC path; PMOS off by default) --------
Rg6   n6   vdd 100k
Rg7   n7   vdd 100k
Rg8   n8   vdd 100k
Rg9   n9   vdd 100k
Rg10  n10  vdd 100k
Rg11  n11  vdd 100k
Rg12  n12  vdd 100k
Rg13  n13  vdd 100k
Rg14  n14  vdd 100k

* -------- Control stimuli for the NMOS tree --------
* n4 and n5 are “decoder” rails; pulses will selectively pull some PMOS gates low.
V4   n4  0  PULSE(0 {VDD} 0       100p 100p 5u 10u)
V5   n5  0  PULSE(0 {VDD} 2.5u    100p 100p 5u 10u)

* -------- Current sources (kept, but neutralize ambiguous ones) --------
* Original: I1 99 3 DC 10uA  -> make node 99 the output and reference to 0
I1   vout  0  DC 10uA

* Original sources to poorly defined nodes -> set to 0 A to avoid breaking DC
I2   0   n8   DC 0A
I3   0   n4   DC 0A
I4   vout n5  DC 0A
I5   vout n9  DC 0A

* -------- Analyses & outputs --------
.OPTION POST=2 NOMOD
.OP
.TRAN 100n 50u
.PROBE V(vout) V(n4) V(n5) V(n6) V(n8) V(n9) V(n10) V(n11) V(n12) V(n13)

.END
