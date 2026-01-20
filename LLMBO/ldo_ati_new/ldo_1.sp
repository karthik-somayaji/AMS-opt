** Copyright: Suming Lai
** Oct. 10, 2011
** Modified by: Honghuang Lin
** Sep. 10, 2015

.PARAM voutac=0 vdddc=1.2 vddac=0 il=0.000000e+00 ib=5u cf=1.600000e-13 cz=2.000000e-13 cc2=4.000000e-13 cc1=1.000000e-12 Rf1=7.800000e+04 Rf2=1.220000e+05

**.AC DEC 200 1.000000e+00 1.000000e+12

.TRAN 1n 16u

.TEMP 25
.OPTION
+    ARTIST=2
+    INGOLD=2
+    MEASOUT=1
+    PARHIER=LOCAL
+    POST
+    PSF=2
+    NOMOD
+        MEASDGT=7
* set the measure output digits

.inc ./param.inc
.INCLUDE "../tech/hspice.include"
.inc ../tech/nfet_perfect.inc
.inc ../tech/pfet_perfect.inc

ib vbias2x my_gnd DC=ib

cz vfb2x vout cz
c2 ampoutpx my_gnd cc2
c1 amp1outpx my_gnd cc1
cc ampoutpx vfb2x cf
r1 vout vfb2x Rf1
r2 vfb2x my_gnd Rf2
iLoad vout 0 PWL 0 0.005 1u 0.005 1.1u 0.105 3u 0.105

v0 my_gnd 0 DC=0 AC=0

vr vref2 0 DC=610e-3 AC=0

vcc my_vdd 0 DC=vdddc AC=vddac

** Mc
xt1 net0237 ampoutpx vout vout pfet l='80e-9+l1*920e-9' w='120e-9+w1*49880e-9' par=1 m=1 nf=200 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0

** Mdb
xt2 net0237 amp1outnx my_gnd my_gnd nfet l='80e-9+l2*920e-9' w='120e-9+w2*49880e-9' par=1 m=1 nf=30 ptwell=1 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M2
xt3 amp1outpx vref2 net0167 net0167 pfet l='80e-9+l3*920e-9' w='120e-9+w3*49880e-9' par=1 ad=1.92e-12 as=2.12e-12 pd=15.84e-6 ps=18.24e-6 m=1 nf=12 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M1
xt4 amp1outnx vfb2x net0167 net0167 pfet l='80e-9+l4*920e-9' w='120e-9+w4*49880e-9' par=1 ad=1.92e-12 as=2.12e-12 pd=15.84e-6 ps=18.24e-6 m=1 nf=12 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M3
xt5 amp1outpx vref2 net0278 my_gnd nfet l='80e-9+l5*920e-9' w='120e-9+w5*49880e-9' par=1 ad=153.6e-15 as=201.6e-15 pd=2.24e-6 ps=3.12e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M4
xt6 amp1outnx vfb2x net0278 my_gnd nfet l='80e-9+l6*920e-9' w='120e-9+w6*49880e-9' par=1 ad=153.6e-15 as=201.6e-15 pd=2.24e-6 ps=3.12e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M1
xt4 amp1outnx vfb2x net0167 net0167 pfet l='80e-9+l7*920e-9' w='120e-9+w7*49880e-9' par=1 ad=1.92e-12 as=2.12e-12 pd=15.84e-6 ps=18.24e-6 m=1 nf=12 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M3
xt5 amp1outpx vref2 net0278 my_gnd nfet l='80e-9+l8*920e-9' w='120e-9+w8*49880e-9' par=1 ad=153.6e-15 as=201.6e-15 pd=2.24e-6 ps=3.12e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M4
xt6 amp1outnx vfb2x net0278 my_gnd nfet l='80e-9+l9*920e-9' w='120e-9+w9*49880e-9' par=1 ad=153.6e-15 as=201.6e-15 pd=2.24e-6 ps=3.12e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M-1
xt7 amp1outpx amp1outpx my_gnd my_gnd nfet l='80e-9+l10*920e-9' w='120e-9+w10*49880e-9' par=1 ad=112e-15 as=147e-15 pd=1.98e-6 ps=2.73e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M-2
xt8 amp1outnx amp1outnx my_gnd my_gnd nfet l='80e-9+l11*920e-9' w='120e-9+w11*49880e-9' par=1 ad=112e-15 as=147e-15 pd=1.98e-6 ps=2.73e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M-3
xt9 amp1outpx amp1outnx my_gnd my_gnd nfet l='80e-9+l12*920e-9' w='120e-9+w12*49880e-9' par=1 ad=105.6e-15 as=138.6e-15 pd=1.94e-6 ps=2.67e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M-4
xt10 amp1outnx amp1outpx my_gnd my_gnd nfet l='80e-9+l13*920e-9' w='120e-9+w13*49880e-9' par=1 ad=105.6e-15 as=138.6e-15 pd=1.94e-6 ps=2.67e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** M5
xt11 ampoutpx amp1outpx my_gnd my_gnd nfet l='80e-9+l14*920e-9' w='120e-9+w14*49880e-9' par=1 ad=352e-15 as=572e-15 pd=2.84e-6 ps=5.44e-6 m=1 nf=2 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** Mp/3
xtp1 vout net0237 my_vdd my_vdd pfet l='80e-9+l15*920e-9' w='120e-9+w15*49880e-9' par=1 m=1 nf=500 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 
xtp2 vout net0237 my_vdd my_vdd pfet l='80e-9+l16*920e-9' w='120e-9+w16*49880e-9' par=1 m=1 nf=500 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 
xtp3 vout net0237 my_vdd my_vdd pfet l='80e-9+l17*920e-9' w='120e-9+w17*49880e-9' par=1 m=1 nf=500 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** Mp'
xtpn vout net0237 my_gnd my_gnd nfet l='80e-9+l18*920e-9' w='120e-9+w18*49880e-9' par=1 ad=117e-15 as=117e-15 pd=1.42e-6 ps=1.42e-6 m=1 nf=1 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=0 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

xtb0 vbias2x vbias2x my_vdd my_vdd pfet l='80e-9+l19*920e-9' w='120e-9+w19*49880e-9' par=1 ad=1.536e-12 as=2.016e-12 pd=10.88e-6 ps=16.08e-6 m=1 nf=4 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

xtb1 net0183 vbias2x my_vdd my_vdd pfet l='80e-9+l20*920e-9' w='120e-9+w20*49880e-9' par=1 ad=1.536e-12 as=2.016e-12 pd=10.88e-6 ps=16.08e-6 m=1 nf=4 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** Mbp1 l='lmlt*960n
xtb2 net0167 vbias2x my_vdd my_vdd pfet l='80e-9+l21*920e-9' w='120e-9+w21*49880e-9' par=1 ad=15.36e-12 as=15.84e-12 pd=108.8e-6 ps=114e-6 m=1 nf=40 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** Mbp2
xtb3 ampoutpx vbias2x my_vdd my_vdd pfet l='80e-9+l22*920e-9' w='120e-9+w22*49880e-9' par=1 ad=2.304e-12 as=2.784e-12 pd=16.32e-6 ps=21.52e-6 m=1 nf=6 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

** Mbn
xtbn0 net0278 vbias1x my_gnd my_gnd nfet l='80e-9+l23*920e-9' w='120e-9+w23*49880e-9' par=1 ad=960e-15 as=1.02e-12 pd=12.4e-6 ps=13.4e-6 m=1 nf=20 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

xtbn1 vbias1x vbias1x my_gnd my_gnd nfet l='80e-9+l24*920e-9' w='120e-9+w24*49880e-9' par=1 ad=192e-15 as=252e-15 pd=2.48e-6 ps=3.48e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

xtbn2 vbias1x vbias1x net0183 net0183 pfet l='80e-9+l25*920e-9' w='120e-9+w25*49880e-9' par=1 ad=1.536e-12 as=2.016e-12 pd=10.88e-6 ps=16.08e-6 m=1 nf=4 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 

*.ALTER
*.param  voutac=1 vddac=0
.OP

.MEAS TRAN iq0 AVG I(vcc) FROM=0u TO=0.1u
.MEAS TRAN iq1 AVG I(vcc) FROM=15.9u TO=16u
.MEAS TRAN uds MIN V(vout) FROM=0u TO=6u
.MEAS TRAN vout0 AVG V(vout) FROM=0n TO=100n
.MEAS TRAN vout1 AVG V(vout) FROM=15.9u TO=16u
.MEAS TRAN voutr AVG V(vout) FROM=15.8u TO=15.9u