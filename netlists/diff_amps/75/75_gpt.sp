* Diff pair with PMOS load + resistive CMFB  (runnable)
* AnalogGenie 75 — per-device W/L and separate CMFB resistors

***** Parameters (X) *****
.param VDD=1.8 
.param ITAIL=200u

* Per-device transistor geometries (defaults keep original symmetry)
.param WN0=10u  LN0=0.5u     * M0 (NMOS, left side)
.param WN1=10u  LN1=0.5u     * M1 (NMOS, right side)
.param WP2=20u  LP2=0.5u     * M2 (PMOS load over VOUT1)
.param WP3=20u  LP3=0.5u     * M3 (PMOS load over VOUT2)

* Separate CMFB resistors (start equal)
.param RCM1=20k
.param RCM2=20k

***** Operating Parameters *****
.param VCM=0.8          * input common-mode
.param VID=10m          * differential amplitude (VIN1 = +VID/2, VIN2 = -VID/2)

***** Power & references *****
VDD  VDD  0  {VDD}
VVSS VSS  0  0           * tie node VSS to ground

***** Inputs *****
VIN1 VIN1 0  {VCM + VID/2}
VIN2 VIN2 0  {VCM - VID/2}

***** Tail current source *****
ISS  IB1  0  DC {ITAIL}  * sinks current from sources of M0/M1

***** Devices *****
* PMOS active loads (sources at VDD, bulks at VDD)
M2  VOUT1 net14 VDD VDD  pmos4  W={WP2} L={LP2}
M3  VOUT2 net14 VDD VDD  pmos4  W={WP3} L={LP3}

* NMOS differential pair (sources share tail node IB1, bulks at VSS)
M0  VOUT1 VIN1 IB1 VSS   nmos4  W={WN0} L={LN0}
M1  VOUT2 VIN2 IB1 VSS   nmos4  W={WN1} L={LN1}

* Resistive CMFB from outputs to PMOS gates (separate values)
R1  VOUT1 net14  {RCM1}
R0  net14  VOUT2 {RCM2}

***** Simple MOS models (portable) *****
.model nmos4 NMOS (LEVEL=1 VTO=0.70 KP=120e-6 LAMBDA=0.04 GAMMA=0.5 PHI=0.6)
.model pmos4 PMOS (LEVEL=1 VTO=-0.70 KP=50e-6  LAMBDA=0.04 GAMMA=0.5 PHI=0.6)

***** Analyses *****
.op
*.ac dec 100 1 1e9
*.tran 1u 200u

.options post=2 nomod
.end