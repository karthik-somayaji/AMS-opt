** Comparator
.inc ./param0.inc
.INCLUDE "../tech/hspice.include"
.inc ../tech/nfet_perfect.inc
.inc ../tech/pfet_perfect.inc

** Circuit Netlist
.GLOBAL vdd!

.TEMP 25
.OPTION
+    ARTIST=2
+    INGOLD=2
+    MEASOUT=1
+    PARHIER=LOCAL
+    POST
+    PSF=2
+    NOMOD
+    MEASDGT=7
+    OPFILE=1

cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd PWL(0,-2,10u,2,20u,-2)
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! vb 5e-6

xnb vb vb 0 0       nfet l='80e-9+l11*920e-9' w='120e-9+w11*49880e-9'

xn5 3 vb 0 0        nfet l='80e-9+l12*920e-9' w='120e-9+w12*49880e-9'
xn1 1 vip 3 0       nfet l='80e-9+l1*920e-9' w='120e-9+w1*49880e-9'
xn2 2 vin 3 0       nfet l='80e-9+l2*920e-9' w='120e-9+w2*49880e-9'
xp3 1 1 vdd! vdd!   pfet l='80e-9+l3*920e-9' w='120e-9+w3*49880e-9'
xp4 2 2 vdd! vdd!   pfet l='80e-9+l4*920e-9' w='120e-9+w4*49880e-9'
xp6 2 1 vdd! vdd!   pfet l='80e-9+l5*920e-9' w='120e-9+w5*49880e-9'
xp7 1 2 vdd! vdd!   pfet l='80e-9+l6*920e-9' w='120e-9+w6*49880e-9'

xp8 out 2 vdd! vdd! pfet l='80e-9+l7*920e-9' w='120e-9+w7*49880e-9'
xp9 4 1 vdd! vdd!   pfet l='80e-9+l8*920e-9' w='120e-9+w8*49880e-9'
xn10 4 4 0 0        nfet l='80e-9+l9*920e-9' w='120e-9+w9*49880e-9'
xn11 out 4 0 0      nfet l='80e-9+l10*920e-9' w='120e-9+w10*49880e-9'

.param vacd=1 vacc=0

.option list=2 node post=2
.option numdgt=10 measdgt=10
.op
.ac dec 100 1 1e9
.measure ac gain find vdb(out) at 1
.measure BW when vdb(out)='gain-3' fall=1
* targ vdb(out)=gain-3 fall=last
.measure ugain_freq when vdb(out)=0 fall=1
.tran 1n 20u
.MEAS TRAN VTP FIND V(vac) WHEN '2*V(out)-2'=V(vac) CROSS=1
.MEAS TRAN VTM FIND V(vac) WHEN '2*V(out)-2'=V(vac) CROSS=2
.MEAS TRAN VHY PARAM='(VTP-VTM)'
.MEAS TRAN OFF PARAM='(VTP+VTM)/2'
.print V(out) V(vac)


.END
