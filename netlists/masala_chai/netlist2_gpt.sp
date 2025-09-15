* ------------------------------------------------------------
* Two-Channel Differential Input Selector (Analog MUX)
* Completed from the provided skeleton for HSPICE
* ------------------------------------------------------------

.option post=2 nomod
.temp 27
.param VDD=1.8  SEL=1        * SEL=1 selects VIN1/VIN2 (Pair A); SEL=0 selects VIN3/VIN4 (Pair B)

* --- Supplies and references ---
VDD   VDD    0   DC {VDD}
VSS   VSS    0   DC 0

* --- Control logic (select one differential pair) ---
* VCONT1 = SEL*VDD ; VCONT2 = (1-SEL)*VDD
VCTRL1 VCONT1 0   DC {SEL*VDD}
VCTRL2 VCONT2 0   DC {(1-SEL)*VDD}

* --- Differential inputs ---
* Make Pair A favor VOUT1 low (VIN1>VIN2) and Pair B favor VOUT2 low (VIN4>VIN3)
VIN1  VIN1   0   DC 0.90
VIN2  VIN2   0   DC 0.60
VIN3  VIN3   0   DC 0.60
VIN4  VIN4   0   DC 0.90

* --- Bias current sink for both tails ---
* Sinks 100 uA from node IB1 to ground
IBIAS IB1    0   DC 100u

* --- Load resistors (unique names + values) ---
RLOAD1 VDD  VOUT1  10k
RLOAD2 VDD  VOUT2  10k

* --- Transistors: drains, gates, sources, bulks ---
* Pair A (selected by VCONT1 via M5)
M0    VOUT1 VIN1  net8  VSS  nmos4  W=4u  L=180n
M1    VOUT2 VIN2  net8  VSS  nmos4  W=4u  L=180n
M5    net8  VCONT1 IB1  VSS  nmos4  W=2u  L=180n   * tail switch for Pair A

* Pair B (selected by VCONT2 via M4)
M3    VOUT1 VIN4  net15 VSS  nmos4  W=4u  L=180n
M2    VOUT2 VIN3  net15 VSS  nmos4  W=4u  L=180n
M4    net15 VCONT2 IB1  VSS  nmos4  W=2u  L=180n   * tail switch for Pair B

* --- Simple NMOS model (Level 1 for demo) ---
.model nmos4 nmos level=1 VTO=0.5 KP=200e-6 LAMBDA=0.02 GAMMA=0.4 PHI=0.7

* --- Analyses ---
.op
.print op V(VOUT1) V(VOUT2) V(net8) V(net15) I(IBIAS)

* To try the other channel, either set .param SEL=0 above, or use an .ALTER:
*.alter
*.param SEL=0

.end