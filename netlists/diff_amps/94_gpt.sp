* Single-ended differential pair with RP feedback (runnable)

**** Parameters ****
.param VDD=1.8
.param RD=20k
.param RP=15k
.param WN=10u  LN=0.5u        * input devices
.param WTAIL=12u LT=0.5u      * tail device
.param VCM=0.80               * input common-mode
.param VID=10m                * differential amplitude
.param VB1=0.90               * tail gate bias

**** Supplies, inputs, and bias ****
VDD   VDD   0   {VDD}
VGND  VSS   0   0
VIN1  VIN1  0   {VCM + VID/2}
VIN2  VIN2  0   {VCM - VID/2}
Vb    VB1   0   {VB1}

**** Devices (D G S B) ****
* Loads to VDD
R0   VOUT1 VDD   {RD}
R1   VOUT2 VDD   {RD}

* Degeneration/feedback resistor from output to common source node
R2   net07  VOUT1  {RP}

* NMOS differential input pair
M0  VOUT1 VIN1 net07 VSS  nmos4  W={WN}   L={LN}
M2  VOUT2 VIN2 net07 VSS  nmos4  W={WN}   L={LN}

* Tail NMOS current source (gate-biased)
M1  net07  VB1  VSS  VSS  nmos4  W={WTAIL} L={LT}

**** Simple MOS model (portable anywhere) ****
.model nmos4 NMOS (LEVEL=1 VTO=0.50 KP=200e-6 LAMBDA=0.04 GAMMA=0.5 PHI=0.6)

**** Analyses ****
.op
*.ac dec 100 1 1e9
*.tran 1u 200u

.options post=2 nomod
.end