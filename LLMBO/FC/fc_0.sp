* Sim
.inc ./param.inc
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
x15 4 4 0 0 nfet        l=l15     w=w15
x12 9 4 0 0 nfet        l=l12     w=w12
x13 7 9 14 vdd! pfet    l=l13     w=w13
x14 14 7 vdd! vdd! pfet l=l14     w=w14
R1 7 9 r1

* Folded Cascode
x1 5 vip 3 0 nfet       l=l1     w=w1
x2 6 vin 3 0 nfet       l=l2     w=w2
x3 3 4 0 0 nfet         l=l3     w=w3
x4 5 7 vdd! vdd! pfet   l=l4     w=w4
x5 6 7 vdd! vdd! pfet   l=l5     w=w5
x6 8 9 5 vdd! pfet      l=l6     w=w6
x7 out 9 6 vdd! pfet    l=l7     w=w7
R2 8 11 r2
x8 11 8 12 0 nfet       l=l8     w=w8
x9 out 8 13 0 nfet      l=l9     w=w9
x10 12 11 0 0 nfet      l=l10     w=w10
x11 13 11 0 0 nfet      l=l11     w=w11

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
