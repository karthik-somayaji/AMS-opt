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

xnb vb vb 0 0       nfet l='90e-9*(1+10*l1)'  w='90e-9*(1+10*l1)*(1+199*w1)'

xn5 3 vb 0 0        nfet l='90e-9*(1+10*l2)'  w='90e-9*(1+10*l2)*(1+199*w2)'
xn1 1 vip 3 0       nfet l='90e-9*(1+10*l3)'  w='90e-9*(1+10*l3)*(1+199*w3)'
xn2 2 vin 3 0       nfet l='90e-9*(1+10*l4)'  w='90e-9*(1+10*l4)*(1+199*w4)'
xp3 1 1 vdd! vdd!   pfet l='90e-9*(1+10*l5)'  w='90e-9*(1+10*l5)*(1+199*w5)'
xp4 2 2 vdd! vdd!   pfet l='90e-9*(1+10*l6)'  w='90e-9*(1+10*l6)*(1+199*w6)'
xp6 2 1 vdd! vdd!   pfet l='90e-9*(1+10*l7)'  w='90e-9*(1+10*l7)*(1+199*w7)'
xp7 1 2 vdd! vdd!   pfet l='90e-9*(1+10*l8)'  w='90e-9*(1+10*l8)*(1+199*w8)'

xp8 out 2 vdd! vdd! pfet l='90e-9*(1+10*l9)'  w='90e-9*(1+10*l9)*(1+199*w9)'
xp9 4 1 vdd! vdd!   pfet l='90e-9*(1+10*l10)'  w='90e-9*(1+10*l10)*(1+199*w10)'
xn10 4 4 0 0        nfet l='90e-9*(1+10*l11)' w='90e-9*(1+10*l11)*(1+199*w11)'
xn11 out 4 0 0      nfet l='90e-9*(1+10*l12)' w='90e-9*(1+10*l12)*(1+199*w12)'

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
