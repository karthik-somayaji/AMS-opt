* Diff pair with PMOS load + resistive CMFB  (runnable)
* AnalogGenie 75

***** Parameters (X) *****
.param VDD=1.8 
.param ITAIL=200u
.param RCM=20k
.param WN=10u  LN=0.5u
.param WP=20u  LP=0.5u

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
M2  VOUT1 net14 VDD VDD  pmos4  W={WP} L={LP}
M3  VOUT2 net14 VDD VDD  pmos4  W={WP} L={LP}

* NMOS differential pair (sources share tail node IB1, bulks at VSS)
M0  VOUT1 VIN1 IB1 VSS   nmos4  W={WN} L={LN}
M1  VOUT2 VIN2 IB1 VSS   nmos4  W={WN} L={LN}

* Resistive CMFB from outputs to PMOS gates
R1  VOUT1 net14  {RCM}
R0  net14  VOUT2 {RCM}

***** Simple MOS models (portable) *****
.model nmos4 NMOS (LEVEL=1 VTO=0.70 KP=120e-6 LAMBDA=0.04 GAMMA=0.5 PHI=0.6)
.model pmos4 PMOS (LEVEL=1 VTO=-0.70 KP=50e-6  LAMBDA=0.04 GAMMA=0.5 PHI=0.6)

***** Analyses *****
.op
*.ac dec 100 1 1e9
*.tran 1u 200u

.options post=2 nomod
.end