**dcdc
.inc ./param0.inc 
.INCLUDE "../tech/hspice.include"
.inc ../tech/nfet_perfect.inc
.inc ../tech/pfet_perfect.inc

.GLOBAL vdd!

*.PARAM IHi=0.5
*.PARAM ILow=0.1
*.PARAM Tstep=3.1e-05

* Independent Sources
V_triag triag_clk 0 PULSE (0 1 50n 50n 0 100n 100n)
Ibias vdd! vb2 DC 800n

* Transistors
x_T_cmp_21 net054 net053 0 0 nfet                 L=(l1*20.0n + 220.0n) W=(w1*10u + 8u)            AD=2.88p AS=3.08p PD=23.76u PS=26.16u NF=18
x_T_cmp_20 net053 triag_clk net054 0 nfet         L=(l2*80.0n + 400n) W=(w2*1.0u + 1.1u)          AD=546f AS=546f PD=4.72u PS=4.72u NF=1
x_T_bias_2 vb vb 0 0 nfet                    L= (l3*100n + 700n)   W=2u                                AD=320f AS=520f PD=2.64u PS=5.04u NF=2
x_T_amp_4 net154 vb 0 0 nfet                L=  (l4*100n + 700n)  W=140u                                AD=22.4p AS=22.6p PD=184.8u PS=187.2u NF=140
x_T_amp_6 net176 vref net154 0 nfet         L= (l5*0.6u + 1u)  W=12u                                        AD=1.92p AS=2.52p PD=13.28u PS=19.68u NF=4
x_T_amp_11 net157 vout net154 0 nfet        L= (l5*0.6u + 1u) W=12u                                        AD=1.92p AS=2.52p PD=13.28u PS=19.68u NF=4
x_T_amp_3 ve vb2 net157 0 nfet              L= (l5*0.6u + 1u) W=16u                                        AD=2.56p AS=3.36p PD=17.28u PS=25.68u NF=4
x_T_cmp_18 out0 ve net054 0 nfet            L= (l6*25n + 60n) W=2.4u                                        AD=336f AS=576f PD=2.96u PS=5.76u NF=2
x_T_cmp_14 out1 out0 0 0 nfet               L= (l6*25n + 60n) W=2.4u                                        AD=336f AS=576f PD=2.96u PS=5.76u NF=2
x_T_bias_5 vb2 vb2 net169 0 nfet            L= (l7*25n + 60n) W=8u                                         AD=1.12p AS=1.92p PD=8.56u PS=16.96u NF=2
x_T_bias_7 net169 vb2 vb 0 nfet             L= (l7*25n + 60n) W=3u                                         AD=720f AS=720f PD=6.48u PS=6.48u NF=1
x_T_cmp_16 vsw out1 0 0 nfet                L= (l8*25n + 60n) W=9.6u                                         AD=1.344p AS=1.584p PD=11.84u PS=14.64u NF=8
x_TSWN net173 vsw 0 0 nfet                  L= (l8*25n + 60n) W=5m                                         AD=700p AS=700.25p PD=6.12m PS=6.1229m NF=4000
x_T_cmp_17 net053 triag_clk net0107 net0107 pfet         L=(l9*80.0n + 400n) W=5.6u                          AD=896f AS=1.176p PD=6.88u PS=10.08u NF=4
x_T_cmp_15 net0107 net053 vdd! vdd! pfet                 L= (l10*20.0n + 220.0n) W=40u                           AD=6.4p AS=6.6p PD=52.8u PS=55.2u NF=40
x_T_amp_8 net176 net176 vdd! vdd! pfet                   L=(l11*20.0n + 300.0n) W=6.6u                           AD=1.056p AS=1.386p PD=7.88u PS=11.58u NF=4
x_T_amp_12 net187 net176 vdd! vdd! pfet                 L= (l11*20.0n + 300.0n) W=6.6u                            AD=1.056p AS=1.386p PD=7.88u PS=11.58u NF=4
x_T_amp_9 ve net154 net187 net187 pfet                  L= (l11*20.0n + 300.0n) W=14u                            AD=2.24p AS=2.94p PD=15.28u PS=22.68u NF=4
x_T_cmp_13 out1 out0 vdd! vdd! pfet                     L= (l12*25n + 60n) W=5.3u                            AD=742f AS=1.272p PD=5.86u PS=11.56u NF=2
x_T_cmp_19 out0 ve net0107 net0107 pfet                 L=(l12*25n + 60n) W=10.8u                            AD=1.512p AS=2.052p PD=11.92u PS=17.72u NF=4
x_T_cmp_10 vsw out1 vdd! vdd! pfet                      L=(l12*25n + 60n) W=26u                            AD=3.64p AS=4.29p PD=28.24u PS=35.14u NF=8
x_TSWP net173 vsw vdd! vdd! pfet                       L=(l12*25n + 60n) W=14m                            AD=1.96n AS=1.9604n PD=15.96m PS=15.9644m NF=7000

* Passive Components
L_L0 net208 vout 400n
C_Cc vout net200 10p
C_C0 net203 0 5u
R_Rc net200 ve 500k
R_R_CESR vout net203 10m
R_R_LESR net173 net208 50m

V_sup vdd! 0 DC=1 AC=0
V_ref vref 0 dc=600.0m ac=0

* Analysis commands
.TRAN 1n 30u
.MEASURE tran overshoot MAX v(vout) FROM=0n TO=30u
.MEASURE tran ripplemax MAX v(vout) FROM=29u TO=30u
.MEASURE tran ripplemin MIN v(vout) FROM=29u TO=30u
.MEASURE tran outavg AVG v(vout) FROM=29u TO=30u
.MEASURE tran pwrcur INTEG i(V_sup) FROM=29u TO=30u
.MEASURE tran pwrout INTEG v(vout)*i(L0) FROM=29u TO=30u
.OPTIONS POST

.END