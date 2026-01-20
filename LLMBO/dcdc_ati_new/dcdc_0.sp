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
x_T_cmp_21 net054 net053 0 0 nfet                 L=80e-9+l1*920e-9 W='120e-9+w1*49880e-6'          
x_T_cmp_20 net053 triag_clk net054 0 nfet         L=80e-9+l2*920e-9 W='120e-9+w2*49880e-6'          
x_T_bias_2 vb vb 0 0 nfet                    L= 80e-9+l3*920e-9   W='120e-9+w3*49880e-6'                              
x_T_amp_4 net154 vb 0 0 nfet                L=  80e-9+l4*920e-9  W='120e-9+w4*49880e-6'                                
x_T_amp_6 net176 vref net154 0 nfet         L= 80e-9+l5*920e-9  W='120e-9+w5*49880e-6'                                    
x_T_amp_11 net157 vout net154 0 nfet        L= 80e-9+l6*920e-9 W='120e-9+w6*49880e-6'                                   
x_T_amp_3 ve vb2 net157 0 nfet              L= 80e-9+l7*920e-9 W='120e-9+w7*49880e-6'                                   
x_T_cmp_18 out0 ve net054 0 nfet            L= 80e-9+l8*920e-9 W='120e-9+w8*49880e-6'                                       
x_T_cmp_14 out1 out0 0 0 nfet               L= 80e-9+l9*920e-9 W='120e-9+w9*49880e-6'                                        
x_T_bias_5 vb2 vb2 net169 0 nfet            L= 80e-9+l10*920e-9 W='120e-9+w10*49880e-6'                                      
x_T_bias_7 net169 vb2 vb 0 nfet             L= 80e-9+l11*920e-9 W='120e-9+w11*49880e-6'                                        
x_T_cmp_16 vsw out1 0 0 nfet                L= 80e-9+l12*920e-9 W='120e-9+w12*49880e-6'                                      
x_TSWN net173 vsw 0 0 nfet                  L= 80e-9+l13*920e-9 W='120e-9+w13*49880e-6'                                         
x_T_cmp_17 net053 triag_clk net0107 net0107 pfet         L= 80e-9+l14*920e-9 W='120e-9+w14*49880e-6'                     
x_T_cmp_15 net0107 net053 vdd! vdd! pfet                 L= 80e-9+l15*920e-9 W='120e-9+w15*49880e-6'                          
x_T_amp_8 net176 net176 vdd! vdd! pfet                   L= 80e-9+l16*920e-9 W='120e-9+w16*49880e-6'                        
x_T_amp_12 net187 net176 vdd! vdd! pfet                 L= 80e-9+l17*920e-9 W='120e-9+w17*49880e-6'                        
x_T_amp_9 ve net154 net187 net187 pfet                  L= 80e-9+l18*920e-9 W='120e-9+w18*49880e-6'                         
x_T_cmp_13 out1 out0 vdd! vdd! pfet                     L= 80e-9+l19*920e-9 W='120e-9+w19*49880e-6'                           
x_T_cmp_19 out0 ve net0107 net0107 pfet                 L= 80e-9+l20*920e-9 W='120e-9+w20*49880e-6'                         
x_T_cmp_10 vsw out1 vdd! vdd! pfet                      L= 80e-9+l21*920e-9 W='120e-9+w21*49880e-6'                        
x_TSWP net173 vsw vdd! vdd! pfet                       L= 80e-9+l22*920e-9 W='120e-9+w22*49880e-6'             

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