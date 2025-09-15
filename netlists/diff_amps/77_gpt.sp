* ------------------------------------------------------------
* Differential Amplifier (from provided sketch)
* NMOS diff pair + resistive loads; PMOS pull-ups biased off
* ------------------------------------------------------------

.option post=2 nomod
.temp 27

* --- Design parameters (easy to tweak) ---
.param VDD=1.8 RD=10k ISS=200u VINCM=0.8 VDIFF=0m
* PMOS bias: set equal to VDD to turn M2/M3 off (|VGS|≈0)
.param VB1={VDD}

* --- Supplies and references ---
VDD   VDD     0     {VDD}
VSS   VSS     0     0
* Tail current source (sinks ISS from node IB1 to ground)
IISS  IB1     0     {ISS}
* PMOS gate bias
VBIAS VB1     0     {VB1}

* --- Differential inputs (DC common-mode with optional small diff) ---
VINP  VIN1    0     DC {VINCM + VDIFF/2}
VINN  VIN2    0     DC {VINCM - VDIFF/2}
*.ac dec 20 1 1e6       * (optional) enable and give AC magnitudes if you want AC gain

* --- Loads ---
R0    VDD   VOUT2  {RD}
R1    VDD   VOUT1  {RD}

* --- Transistors: D  G    S    B   model   size ---
* NMOS differential pair
M0    VOUT1 VIN1  IB1  VSS  nmos4  W=4u  L=180n
M1    VOUT2 VIN2  IB1  VSS  nmos4  W=4u  L=180n

* PMOS pull-ups (kept off by VB1=VDD; lower VB1 to ~1.0–1.2 V to activate)
M2    VOUT1 VB1   VDD  VDD  pmos4  W=3u  L=180n
M3    VOUT2 VB1   VDD  VDD  pmos4  W=3u  L=180n

* --- Simple device models (demo-friendly) ---
.model nmos4 nmos level=1 VTO=0.5 KP=200e-6 LAMBDA=0.02 GAMMA=0.4 PHI=0.7
.model pmos4 pmos level=1 VTO=-0.5 KP=100e-6 LAMBDA=0.02 GAMMA=0.4 PHI=0.7

* --- Analyses / outputs ---
.op
.print op V(VOUT1) V(VOUT2) V(VIN1) V(VIN2) I(IISS)
.tran 0.1u 50u
.probe V(VOUT1) V(VOUT2)

.end