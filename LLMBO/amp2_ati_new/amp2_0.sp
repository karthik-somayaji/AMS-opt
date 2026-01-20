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
ib vdd! vb 5e-6

xp1 node1 node1 vdd! vdd! pfet l='80e-9+l1*920e-9' w='120e-9+w1*49880e-9'
xp2 out1 node1 vdd! vdd!  pfet l='80e-9+l2*920e-9' w='120e-9+w2*49880e-9'
xn1 node1 vip node2 0     nfet l='80e-9+l3*920e-9' w='120e-9+w3*49880e-9'
xn2 out1 vin node2 0      nfet l='80e-9+l4*920e-9' w='120e-9+w4*49880e-9'
xnb vb vb 0 0             nfet l='80e-9+l5*920e-9' w='120e-9+w5*49880e-9'
xnc node2 vb 0 0          nfet l='80e-9+l6*920e-9' w='120e-9+w6*49880e-9'
xpo out out1 vdd! vdd!    pfet l='80e-9+l7*920e-9' w='120e-9+w7*49880e-9'
xno out vb 0 0            nfet l='80e-9+l8*920e-9' w='120e-9+w8*49880e-9'
cc out outm 'pwr(10,c1*4-2)*1e-12'
rz outm out1 'pwr(10,r1*4-2)*1e3'

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
