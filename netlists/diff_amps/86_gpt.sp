*------------------------------------------------------------
* Differential pair with PMOS active loads and resistive
* common-mode feedback (matches the provided schematic)
* AnalogGenie 86
*------------------------------------------------------------
.OPTION POST BRIEF NOMOD
.TEMP 27

*---------------- Parameters ----------------
.PARAM  VDD=1.8  VCM=0.80  VDIFF=20m  VB=0.90
.PARAM  LN=0.18u  LP=0.18u  WN=3u  WP=6u  WTAIL=6u
.PARAM  RL=100k

*---------------- Supplies & Sources --------
VVDD   VDD   0   DC 'VDD'
VVSS   VSS   0   0

* differential inputs toggling around VCM
VIN1   VIN1  0   PWL( 0     'VCM' \
                     1u    'VCM + VDIFF/2' \
                     2u    'VCM - VDIFF/2' \
                     3u    'VCM + VDIFF/2' \
                     4u    'VCM - VDIFF/2' )
VIN2   VIN2  0   PWL( 0     'VCM' \
                     1u    'VCM - VDIFF/2' \
                     2u    'VCM + VDIFF/2' \
                     3u    'VCM - VDIFF/2' \
                     4u    'VCM + VDIFF/2' )
VBIAS  VB1   0   DC 'VB'

*---------------- Transistors ---------------
* NMOS input pair
M0  VOUT1 VIN1 net08 VSS nmos4 W='WN'   L='LN'
M2  VOUT2 VIN2 net08 VSS nmos4 W='WN'   L='LN'

* NMOS tail current source
M1  net08  VB1 VSS  VSS nmos4 W='WTAIL' L='LN'

* PMOS active loads; gates tied to mid node of R-string
M3  VOUT1 net010 VDD VDD pmos4 W='WP'   L='LP'
M4  VOUT2 net010 VDD VDD pmos4 W='WP'   L='LP'

*---------------- Resistor network ----------
R0  net010 VOUT1  'RL'
R1  VOUT2  net010 'RL'

*---------------- Models (Level-1) ----------
.MODEL nmos4 NMOS LEVEL=1 VTO=0.50 KP=200u GAMMA=0.40 PHI=0.70 LAMBDA=0.02
.MODEL pmos4 PMOS LEVEL=1 VTO=-0.50 KP=100u GAMMA=0.40 PHI=0.70 LAMBDA=0.02

*---------------- Analyses ------------------
.OP
.TRAN 0.1u 5u
*.AC DEC 20 1 100MEG           * (optional small-signal run)

*---------------- Outputs to save -----------
.PROBE V(VOUT1) V(VOUT2) V(net010) I(VVDD)

.END