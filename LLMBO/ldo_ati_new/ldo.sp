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

.inc ./param0.inc
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
xt1 net0237 ampoutpx vout vout pfet l='8.000000e-08' w=8.000000e-05 par=1 m=1 nf=200 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt0 vmlt=vmlt0 omlt=omlt0

** Mdb
xt2 net0237 amp1outnx my_gnd my_gnd nfet l='8.000000e-08' w=2.700000e-05 par=1 m=1 nf=30 ptwell=1 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt1 vmlt=vmlt1 omlt=omlt1

** M2
xt3 amp1outpx vref2 net0167 net0167 pfet l='160e-9' w=12e-6 par=1 ad=1.92e-12 as=2.12e-12 pd=15.84e-6 ps=18.24e-6 m=1 nf=12 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt2 vmlt=vmlt2 omlt=omlt2

** M1
xt4 amp1outnx vfb2x net0167 net0167 pfet l='160e-9' w=12e-6 par=1 ad=1.92e-12 as=2.12e-12 pd=15.84e-6 ps=18.24e-6 m=1 nf=12 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt3 vmlt=vmlt3 omlt=omlt3

** M3
xt5 amp1outpx vref2 net0278 my_gnd nfet l='160e-9' w=960e-9 par=1 ad=153.6e-15 as=201.6e-15 pd=2.24e-6 ps=3.12e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt4 vmlt=vmlt4 omlt=omlt4

** M4
xt6 amp1outnx vfb2x net0278 my_gnd nfet l='160e-9' w=960e-9 par=1 ad=153.6e-15 as=201.6e-15 pd=2.24e-6 ps=3.12e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt5 vmlt=vmlt5 omlt=omlt5

** M1

** M3

** M4

** M-1
xt7 amp1outpx amp1outpx my_gnd my_gnd nfet l='160e-9' w=700e-9 par=1 ad=112e-15 as=147e-15 pd=1.98e-6 ps=2.73e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt6 vmlt=vmlt6 omlt=omlt6

** M-2
xt8 amp1outnx amp1outnx my_gnd my_gnd nfet l='160e-9' w=700e-9 par=1 ad=112e-15 as=147e-15 pd=1.98e-6 ps=2.73e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt7 vmlt=vmlt7 omlt=omlt7

** M-3
xt9 amp1outpx amp1outnx my_gnd my_gnd nfet l='160e-9' w=660e-9 par=1 ad=105.6e-15 as=138.6e-15 pd=1.94e-6 ps=2.67e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt8 vmlt=vmlt8 omlt=omlt8

** M-4
xt10 amp1outnx amp1outpx my_gnd my_gnd nfet l='160e-9' w=660e-9 par=1 ad=105.6e-15 as=138.6e-15 pd=1.94e-6 ps=2.67e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt9 vmlt=vmlt9 omlt=omlt9

** M5
xt11 ampoutpx amp1outpx my_gnd my_gnd nfet l='120e-9' w=2.2e-6 par=1 ad=352e-15 as=572e-15 pd=2.84e-6 ps=5.44e-6 m=1 nf=2 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt10 vmlt=vmlt10 omlt=omlt10

** Mp/3
xtp1 vout net0237 my_vdd my_vdd pfet l='80e-9' w=4.500000e-04 par=1 m=1 nf=500 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt11 vmlt=vmlt11 omlt=omlt11
xtp2 vout net0237 my_vdd my_vdd pfet l='80e-9' w=4.500000e-04 par=1 m=1 nf=500 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt11 vmlt=vmlt11 omlt=omlt11
xtp3 vout net0237 my_vdd my_vdd pfet l='80e-9' w=4.500000e-04 par=1 m=1 nf=500 dtemp=0 rgatemod=0 sa=240e-9 sb=240e-9 sd=280e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt11 vmlt=vmlt11 omlt=omlt11

** Mp'
xtpn vout net0237 my_gnd my_gnd nfet l='90e-9' w=450e-9 par=1 ad=117e-15 as=117e-15 pd=1.42e-6 ps=1.42e-6 m=1 nf=1 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=0 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt12 vmlt=vmlt12 omlt=omlt12

xtb0 vbias2x vbias2x my_vdd my_vdd pfet l='960e-9' w=9.6e-6 par=1 ad=1.536e-12 as=2.016e-12 pd=10.88e-6 ps=16.08e-6 m=1 nf=4 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt13 vmlt=vmlt13 omlt=omlt13

xtb1 net0183 vbias2x my_vdd my_vdd pfet l='960e-9' w=9.6e-6 par=1 ad=1.536e-12 as=2.016e-12 pd=10.88e-6 ps=16.08e-6 m=1 nf=4 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt14 vmlt=vmlt14 omlt=omlt14

** Mbp1 l='lmlt*960n
xtb2 net0167 vbias2x my_vdd my_vdd pfet l='960e-9' w=96e-6 par=1 ad=15.36e-12 as=15.84e-12 pd=108.8e-6 ps=114e-6 m=1 nf=40 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt15 vmlt=vmlt15 omlt=omlt15

** Mbp2
xtb3 ampoutpx vbias2x my_vdd my_vdd pfet l='960e-9' w=14.4e-6 par=1 ad=2.304e-12 as=2.784e-12 pd=16.32e-6 ps=21.52e-6 m=1 nf=6 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt16 vmlt=vmlt16 omlt=omlt16

** Mbn
xtbn0 net0278 vbias1x my_gnd my_gnd nfet l='480e-9' w=6e-6 par=1 ad=960e-15 as=1.02e-12 pd=12.4e-6 ps=13.4e-6 m=1 nf=20 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt17 vmlt=vmlt17 omlt=omlt17

xtbn1 vbias1x vbias1x my_gnd my_gnd nfet l='480e-9' w=1.2e-6 par=1 ad=192e-15 as=252e-15 pd=2.48e-6 ps=3.48e-6 m=1 nf=4 ptwell=1 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt18 vmlt=vmlt18 omlt=omlt18

xtbn2 vbias1x vbias1x net0183 net0183 pfet l='480e-9' w=9.6e-6 par=1 ad=1.536e-12 as=2.016e-12 pd=10.88e-6 ps=16.08e-6 m=1 nf=4 dtemp=0 rgatemod=0 sa=260e-9 sb=260e-9 sd=320e-9 panw1=0 panw2=0 panw3=0 panw4=0 panw5=0 panw6=0 panw7=0 panw8=0 panw9=0 panw10=0 lmlt=lmlt19 vmlt=vmlt19 omlt=omlt19

*.ALTER
*.param0  voutac=1 vddac=0
.OP

.MEAS TRAN iq0 AVG I(vcc) FROM=0u TO=0.1u
.MEAS TRAN iq1 AVG I(vcc) FROM=15.9u TO=16u
.MEAS TRAN uds MIN V(vout) FROM=0u TO=6u
.MEAS TRAN vout0 AVG V(vout) FROM=0n TO=100n
.MEAS TRAN vout1 AVG V(vout) FROM=15.9u TO=16u
.MEAS TRAN voutr AVG V(vout) FROM=15.8u TO=15.9u