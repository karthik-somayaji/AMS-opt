* Diff pair with PMOS common-gate loads + source degeneration (runnable)
* AnalogGenie 84

**** Parameters ****
.param VDD=1.8
.param RLOAD=20k
.param WN=10u  LN=0.5u
.param WP=20u  LP=0.5u

* Bias & stimulus
.param VCM=0.80         * input common-mode
.param VID=10m          * small differential input
.param VB1=0.90         * tail NMOS gate bias
.param VB2=0.50         * PMOS load gate bias (sets VSGp via RLOAD drop)

**** Supplies and sources ****
VDD   VDD   0   {VDD}
VGND  VSS   0   0
VIN1  VIN1  0   {VCM + VID/2}
VIN2  VIN2  0   {VCM - VID/2}
Vb1   VB1   0   {VB1}
Vb2   VB2   0   {VB2}

**** Devices (node order: D G S B) ****
* NMOS differential pair
M0  VOUT1 VIN1 net08 VSS  nmos4  W={WN} L={LN}
M2  VOUT2 VIN2 net08 VSS  nmos4  W={WN} L={LN}

* NMOS tail device (M5 in the figure)
M1  net08  VB1  VSS  VSS  nmos4  W={WN} L={LN}

* PMOS common-gate loads with source degeneration
M3  VOUT1 VB2  net014 net014 pmos4 W={WP} L={LP}
M4  VOUT2 VB2  net013 net013 pmos4 W={WP} L={LP}

* Resistive source degeneration to VDD
R0  VDD   net014  {RLOAD}
R1  VDD   net013  {RLOAD}

**** Simple portable MOS models ****
.model nmos4 NMOS (LEVEL=1 VTO=0.70 KP=120e-6 LAMBDA=0.04 GAMMA=0.5 PHI=0.6)
.model pmos4 PMOS (LEVEL=1 VTO=-0.70 KP=50e-6  LAMBDA=0.04 GAMMA=0.5 PHI=0.6)

**** Analyses ****
.op
*.ac dec 100 1 1e9
*.tran 1u 200u

.options post=2 nomod
.end