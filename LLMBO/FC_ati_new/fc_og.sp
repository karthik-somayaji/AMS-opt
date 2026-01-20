
* Sim
.inc ./param0.inc
** Differential amplifier for process variation analysis
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
+    PSF=2
+	 OPFILE=1

cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! 4 5e-6

* Bias
x15 4 4 0 0 nfet        l='80e-9+l15*920e-9'     w='120e-9+w15*49880e-9'
x12 9 4 0 0 nfet        l='80e-9+l12*920e-9'     w='120e-9+w12*49880e-9*1.5'
x13 7 9 14 vdd! pfet    l='80e-9+l13*920e-9'    w='120e-9+w13*49880e-9'
x14 14 7 vdd! vdd! pfet l='80e-9+l14*920e-9'     w='120e-9+w14*49880e-9'
R1 7 9 '1e4*(1+9*r1)'

* Folded Cascode
x1 5 vip 3 0 nfet       l='80e-9+l1*920e-9'     w='120e-9+w1*49880e-9'
x2 6 vin 3 0 nfet       l='80e-9+l2*920e-9'     w='120e-9+w2*49880e-9'
x3 3 4 0 0 nfet         l='80e-9+l3*920e-9'     w='120e-9+w3*49880e-9*2'
x4 5 7 vdd! vdd! pfet   l='80e-9+l4*920e-9'     w='120e-9+w4*49880e-9'
x5 6 7 vdd! vdd! pfet   l='80e-9+l5*920e-9'     w='120e-9+w5*49880e-9'
x6 8 9 5 vdd! pfet      l='80e-9+l6*920e-9'     w='120e-9+w6*49880e-9'
x7 out 9 6 vdd! pfet    l='80e-9+l7*920e-9'     w='120e-9+w7*49880e-9'
R2 8 11 '1e4*(1+9*r2)'
x8 11 8 12 0 nfet       l='80e-9+l8*920e-9'     w='120e-9+w8*49880e-9'
x9 out 8 13 0 nfet      l='80e-9+l9*920e-9'     w='120e-9+w9*49880e-9'
x10 12 11 0 0 nfet      l='80e-9+l10*920e-9'    w='120e-9+w10*49880e-9'
x11 13 11 0 0 nfet      l='80e-9+l11*920e-9'    w='120e-9+w11*49880e-9'


** commands
.option list=2 node post=2
.option numdgt=10 measdgt=10
.op
.ac dec 100 1 1e9
.measure ac gain find vdb(out) at=1
.param vacd=0 vacc=1
.alter
.param vacd=1 vacc=0
.measure ac ugain_freq when vdb(out)=0 fall=1
.measure ac phase_margin MIN vp(out) from=1 to=ugain_freq
.END
