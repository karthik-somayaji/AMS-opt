
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
x15 4 4 0 0 nfet        l='90e-9*(1+10*l15)'     w='90e-9*(1+10*l15)*(1+99*w15)'
x12 9 4 0 0 nfet        l='90e-9*(1+10*l12)'     w='90e-9*(1+10*l12)*(1+99*w12)*1.5'
x13 7 9 14 vdd! pfet    l='90e-9*(1+10*l13)'    w='90e-9*(1+10*l13)*(1+99*w13)'
x14 14 7 vdd! vdd! pfet l='90e-9*(1+10*l14)'     w='90e-9*(1+10*l14)*(1+99*w14)'
R1 7 9 '1e4*(1+9*r1)'

* Folded Cascode
x1 5 vip 3 0 nfet       l='90e-9*(1+10*l1)'     w='90e-9*(1+10*l1)*(1+99*w1)'
x2 6 vin 3 0 nfet       l='90e-9*(1+10*l2)'     w='90e-9*(1+10*l2)*(1+99*w2)'
x3 3 4 0 0 nfet         l='90e-9*(1+10*l3)'     w='90e-9*(1+10*l3)*(1+99*w3)*2'
x4 5 7 vdd! vdd! pfet   l='90e-9*(1+10*l4)'     w='90e-9*(1+10*l4)*(1+99*w4)'
x5 6 7 vdd! vdd! pfet   l='90e-9*(1+10*l5)'     w='90e-9*(1+10*l5)*(1+99*w5)'
x6 8 9 5 vdd! pfet      l='90e-9*(1+10*l6)'     w='90e-9*(1+10*l6)*(1+99*w6)'
x7 out 9 6 vdd! pfet    l='90e-9*(1+10*l7)'     w='90e-9*(1+10*l7)*(1+99*w7)'
R2 8 11 '1e4*(1+9*r2)'
x8 11 8 12 0 nfet       l='90e-9*(1+10*l8)'     w='90e-9*(1+10*l8)*(1+99*w8)'
x9 out 8 13 0 nfet      l='90e-9*(1+10*l9)'     w='90e-9*(1+10*l9)*(1+99*w9)'
x10 12 11 0 0 nfet      l='90e-9*(1+10*l10)'    w='90e-9*(1+10*l10)*(1+99*w10)'
x11 13 11 0 0 nfet      l='90e-9*(1+10*l11)'    w='90e-9*(1+10*l11)*(1+99*w11)'


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
