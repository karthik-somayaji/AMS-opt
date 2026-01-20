* Path, Component, Release: cmos9sf/rel/HSPICE/models/pfet.inc, 9sf_models, mod_cmos9sf
* CMVC Revision: 1.9 06/04/05 10:47:28
***********************************************************************
*
*   Licensed Materials - Property of IBM
*
*  
*   IBM CONFIDENTIAL
*   (C) 2006 IBM Corporation 
*
*   US Government Users Restricted Rights - Use, duplication
*   or disclosure restricted by GSA ADP Schedule contract with
*   IBM Corporation.
************************************************************************
.subckt pfet 1 2 3 4 
+         ngcon=1  ring=0  cwire=0
+         rf_rsub=1 diffca=1 rgatemod=0 rbodymod=0
+       psw_acv_sign=1 rdc=0 rsc=0
+       w=0u l=0u as=0p ad=0p ps=0u pd=0u
+       nrd=0 nrs=0 dtemp=0 nf=1 nwproxs=0
+       panw1=0p panw2=0p panw3=0p panw4=0p panw5=0p
+       panw6=0p panw7=0p panw8=0p panw9=0p panw10=0p
+       sa=0 sb=0 sd=0 ptwell=1 par=1
+       plnest=0 plorient=0 pld200=1 pwd100=-1
+       pdevdops = 1 pdevlgeos= 1 pdevwgeos= 1 p_vta=0
*
* LP
.param lp = 'l-(2*lint_pfet_n)'
* Chip Mean Length Calculations
.param     lchiptola='(lp*0.073+0.0014e-6)'
.param     lchiptol= 'min(lchiptola,0.019e-6)'
* ACV Length Calculations
.param     laclva  = '(lp*0.051-0.0032e-6)'
.param     laclv_1 = 'min(laclva,0.009e-6)'
*************  ACV Calculations  **********

+ aclv_factor_1           = '(aclv_fet_ni*aclv_fet_ni)*((pc_nest_f==1)||(pc_nest_f==3))*(plnest!=0)'
+ aclv_factor_2           = '(aclv_fet_or*aclv_fet_or)*((pc_orient_f==1) ||(pc_orient_f==3))*(plorient!=0)'
+ aclv_factor_3           = '(aclv_fet_dist*aclv_fet_dist)*((pc_dist_f==1)||(pc_dist_f==3))*(pld200!=0)'
+ aclv_factor_4           = '(aclv_fet_ran*aclv_fet_ran)'
+ aclv_factor_5           = 'sqrt(aclv_factor_1 + aclv_factor_2 + aclv_factor_3 + aclv_factor_4)+(laclv_1*len_dep)'
+ aclv_factor_6           = '(aclv_fet_ni)*((pc_nest_f==1)||(pc_nest_f==3))*(plnest!=0)'
+ aclv_factor_7           = '(aclv_fet_or)*((pc_orient_f==1)||(pc_orient_f==3))*(plorient!=0)'
+ aclv_factor_8           = '(aclv_fet_dist)*((pc_dist_f==1)||(pc_dist_f==3))*(pld200!=0)'
+ aclv_factor_9           = '(aclv_factor_6+aclv_factor_7+aclv_factor_8+aclv_fet_ran+(laclv_1*len_dep))'
+ aclv_factor             = '((aclv_factor_9==0)+aclv_factor_5)/((aclv_factor_9==0)+aclv_factor_9)'
* calculate the rss factor*/
.param   a_plnest= limit(-0.5,0.5)	$this gives a random selection of nested or isolated
+   p_plnest=a_plnest
+   sel_plnest='((plnest==1)||((p_plnest*plnest)==1))'
+   a_plorient= limit(-0.5,0.5)	$this gives a random selection of the vert or horz orientation acv
+   p_plorient=a_plorient
+   sel_plorient='((plorient==1)||((p_plorient*plorient)==1))'
+   a_pld200=limit(-0.5,0.5)		$this gives a random selection of dist in or out  acv value
+   p_pld200=a_pld200
+   sel_pldist='((pld200==1)||((p_pld200*pld200)==1))'
+   aclv_ni_sws='(mc_sw+((mc_fet_mm_f!=1)&&(pc_nest==2))/aclv_factor + (cor_acv_correlated/3)*(1-mc_fet_mm_f)*(pc_nest!=2))'
+   aclv_or_sws='(mc_sw +((mc_fet_mm_f!=1)&&(pc_orient==2))/aclv_factor +(cor_acv_correlated/3)*(1-mc_fet_mm_f)*(pc_orient!=2))'
+   aclv_dist_sws='(mc_sw+((mc_fet_mm_f!=1)&&(pc_dist==2))/aclv_factor + (cor_acv_correlated/3)*(1-mc_fet_mm_f)*(pc_dist!=2))'

******local dist***********
.param   aclv_ran=agauss(0,1,3)
+        acv_correlated='cnr_acv_correlated/3'
+        acv_uncorrelated='psw_acv_sign * cnr_acv_uncorrelated/3'
+        aclv_c='cnr_pc/3'
+        acwv_ran=agauss(0,1,3)
+        acwv_c='cnr_rx/3'
+        acwv_rxcrit_mc=agauss(0,1,3)
***/*local distributions used for isolated, l-out200,w-out100*/
**this code was in the skew.file for 8rf moved to subckt to use local dist
.param
+   aclv_ni = '(((fet_aclv_nest*(sel_plnest))+(fet_aclv_iso*(1-(sel_plnest))))*(plnest==-1)*(pc_nest!=2)+(fet_aclv_nest *(plnest==1))+ (fet_aclv_iso * (plnest==2)))*aclv_ni_sws'
+   aclv_or = '(((fet_aclv_v*(sel_plorient))+(fet_aclv_h*(1-(sel_plorient))))*(plorient==-1)*(pc_orient!=2)+(fet_aclv_v *(plorient==1))+ (fet_aclv_h * (plorient==2)))*aclv_or_sws'
+   aclv_dist = '(((fet_aclv_in*(sel_pldist))+(fet_aclv_out*(1-(sel_pldist))))*(pld200==-1)*(pc_dist!=2)+(fet_aclv_in *(pld200==1))+ (fet_aclv_out * (pld200==2)))*aclv_dist_sws'
*  rss_func   = aclv_factor_5*aclv_factor_5*/
.param   rss_func   = '0.007e-6+laclv_1'
.param   systematic = '(pow(aclv_fet_ni,2)+pow(aclv_fet_or,2)+pow(aclv_fet_dist,2))'
.param aclvlenf='((len_dep==1)&&(aclv_fet_ran !=0)) * sqrt(rss_func*rss_func-systematic)/(aclv_fet_ran+(aclv_fet_ran == 0))+((aclv_fet_ran ==0)||(len_dep!=1))'
.param randomf = 'aclvlenf *((aclv_ran)*(mc_fet_mm_f))'
****Lint*****
.param   xc_lint_1='(aclv_ni*aclv_fet_ni)+(aclv_or*aclv_fet_or)+ laclv_1*(acv_correlated*(mc_sw == 0)*len_dep)'
.param   xc_lint_2='((aclv_dist*aclv_fet_dist) + (((acv_uncorrelated*(mc_sw == 0)) + randomf)*aclv_fet_ran) + (aclv_factor_9*(mc_global == 2)))'
.param   xx_lint_3='(mc_fet_mm_f+(aclv_factor*(1-(mc_fet_mm_f))))'
.param   xx_lint_4='((mc_global==2)*(-x_lint_pfet)*(mc_sw)+(mc_global!=2))'
.param   xc_lint_fet='((xc_lint_1+xc_lint_2)*xx_lint_3)*xx_lint_4'
.param   lintchp2    = 'lint_tol_pfet*x_lint_pfet + lint_tol_pfet/2*x_lint_pfet1 '
.param   lintchp3    = '(abs(lchiptol/(lint_tol_pfet*2)))*(len_dep == 1)+ (len_dep != 1)'
.param   lintchpf  = 'lintchp2*lintchp3'

******/*final lint   */
.param   lint_pfet ='lint_pfet_n+0.5*xc_lint_fet*fet_geo_mm*pdevlgeos+  lintchpf + lint_pfet_add'
.param   xc_wint_1='(((acv_uncorrelated*(mc_sw==0))+(acwv_ran*(mc_fet_mm_f))+(mc_global==2))*acwv_fet_ran)'
.param   xx_wint_2='((mc_global==2)*(-x_wint_pfet)*(mc_sw)+(mc_global!=2))'
.param   xc_wint_pfet='xc_wint_1*xx_wint_2'
.param   wint_pfet='wint_pfet_n-0.5*xc_wint_pfet*fet_geo_mm*pdevwgeos+(wint_pfet_n-wint_pfet_b)*x_wint_pfet + wint_pfet_add'
.param      dlc_pfet   = 'dlcslope_pfet*lint_pfet + ladj_pfet'
.param      dwc_pfet   = 'wint_pfet + wadj_pfet'
.param      dwj_pfet   = 'wint_pfet + wjadj_pfet'
****************************************************************
*********************** NWELL PROX DEFAULT ****************************
.param p_logic = '(((panw1 != 0.0) + (panw2 != 0.0) + (panw3 != 0.0) + (panw4 != 0.0) + (panw5 != 0.0) + (panw6 != 0.0) + (panw7 != 0.0) + (panw8 != 0.0) + (panw9 != 0.0) + (panw10 != 0.0)) >= 1)'
.param proxfinal = '(((gwells==2) && (p_logic==0)) ? 1 : 0 )'
.param panw1f =  '((proxfinal==0)*panw1)* (gwells > 0)' 
.param panw2f = '(((proxfinal==0)*panw2) + (((band2-2*band1)<(w/nf)) * l *(band2-2*band1) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw3f = '(((proxfinal==0)*panw3)   +  (((band3-2*band1)<(w/nf)) * l *(band3-band2) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw4f = '(((proxfinal==0)*panw4)   +  (((band4-2*band1)<(w/nf)) * l *(band4-band3) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw5f = '(((proxfinal==0)*panw5)   +  (((band5-2*band1)<(w/nf)) * l *(band5-band4) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw6f = '(((proxfinal==0)*panw6)   +  (((band6-2*band1)<(w/nf)) * l *(band6-band5) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw7f = '(((proxfinal==0)*panw7)   +  (((band7-2*band1)<(w/nf)) * l *(band7-band6) * nf *(proxfinal==1))) * (gwells > 0)'
.param panw8f = '(((proxfinal==0)*panw8)   +  (((band8-2*band1)<(w/nf)) * l *(band8-band7) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw9f = '(((proxfinal==0)*panw9)   +  (((band9-2*band1)<(w/nf)) * l *(band9-band8) * nf *(proxfinal==1))) * (gwells > 0)' 
.param panw10f = '(((proxfinal==0)*panw10)   +  (((band10-2*band1)<(w/nf)) * l *(band10-band9) * nf *(proxfinal==1))) * (gwells > 0)' 
*********Device Parameters for 9SF********
.param     proxpre_tw = 0.014
+          proxlen_tw = 0.4e-06
+          proxminlen_tw = 0.4e-06
+          proxpre_mid = '(0.014 + proxpre_tw)/2'
+          proxlen_mid = '(0.4e-06 + proxlen_tw)/2'
+          proxminlen_mid = '(0.4e-06 + proxminlen_tw)/2'
+          proxpre= '(0.014*(ptwell==0)) +(proxpre_tw * (ptwell==1)) + (proxpre_mid * (ptwell == 2))'
+          proxlen= '(0.4e-06*(ptwell==0)) +(proxlen_tw * (ptwell==1)) + (proxlen_mid * (ptwell == 2))'
+          proxminlen= '(0.4e-06*(ptwell==0)) +(proxminlen_tw * (ptwell==1)) + (proxminlen_mid * (ptwell == 2))'
+          rxwparm1= 0.12
+          rxwparm2= 0.3e-06
+          rxwparm3 = 8.0
+          rxwparm4 = 0.1e-06
******************************************************************
+ nw1b1   = 'panw1f * exp(-pow(((band1-proxminlen)/proxlen),2))'
+ nw2b2   = 'panw2f * exp(-pow(((band2-proxminlen)/proxlen),2))'
+ nw3b3   = 'panw3f * exp(-pow(((band3-proxminlen)/proxlen),2))'
+ nw4b4   = 'panw4f * exp(-pow(((band4-proxminlen)/proxlen),2))'
+ nw5b5   = 'panw5f * exp(-pow(((band5-proxminlen)/proxlen),2))'
+ nw6b6   = 'panw6f * exp(-pow(((band6-proxminlen)/proxlen),2))'
+ nw7b7   = 'panw7f * exp(-pow(((band7-proxminlen)/proxlen),2))'
+ nw8b8   = 'panw8f * exp(-pow(((band8-proxminlen)/proxlen),2))'
+ nw9b9   = 'panw9f * exp(-pow(((band9-proxminlen)/proxlen),2))'
+ nw10b10 = 'panw10f * exp(-pow(((band10-proxminlen)/proxlen),2))'
+ lenratio = '1 + (atan(1.0 * (l/0.08e-06)-1)/3.14)'
+ prox_term1 = 'lenratio*proxpre/(l*w)*(nw1b1+nw2b2+nw3b3+nw4b4+nw5b5)'
+ prox_term2 = 'lenratio*proxpre/(l*w)*(nw6b6+nw7b7+nw8b8+nw9b9+nw10b10)'
+ prox_term  = 'prox_term1 + prox_term2'
*********************STI Stress Model**********************************
.param sti_logic = '(((sa != 0.0) + (sb != 0.0) + (sd != 0.0)) >=1)'
.param sa_f = '(sti_logic==0)*p_sti + sa*(sti_logic==1)*(gstis>0)'
.param sb_f = '(sti_logic==0)*p_sti + sb*(sti_logic==1)*(gstis>0)'
.param sd_f = 'sd*(sti_logic==1)*(gstis>0)'
*************Device parameters for 9SF**********************************
+          p_dlnom = '2*lint_pfet'
+          lvsat_pfet = -5578.58 
+          eta0_pfet = 0 
+          leta0_pfet = 0.018404 
+          lrdsw_pfet = -20.677 
+          lu0_pfet = -0.172659 
+          wu0_pfet = 0.457392 
+          lk1_pfet = -0.00878444 
+          lnfactor_pfet = -0.048768 
****************************************************************************
+          p_leff = 'l*1e+6 - p_dlnom*1e+6'
+          pwpar  = 'w*1e+6*par'
+          pmm_area = '1/sqrt((p_leff)*pwpar)'
*---------NEW version of mismatch-------------------------*/
* calculate tracking multipliers for:                     */
*  u0                                                     */
*  vsat                                                   */
*  rdsw                                                   */
*  eta0                                                   */
*  vth0                                                   */
*  k1                                                     */
*  nfactor                                                */
*  dvtp0                                                  */
*  dvtp1                                                  */
*  pdiblc2                                                */
.param p_mismatchd1  = agauss(0,3,3)
.param mismatchd1  = p_mismatchd1
.param p_mismatchd2  = agauss(0,3,3)
.param mismatchd2  = p_mismatchd2
.param p_mismatchd3  = agauss(0,3,3)
.param mismatchd3  = p_mismatchd3
.param p_mismatchd4  = agauss(0,3,3)
.param mismatchd4  = p_mismatchd4
.param p_mismatchd5  = agauss(0,3,3)
.param mismatchd5  = p_mismatchd5
* fet_dop_mm turns on matching calculations         sw setting       */
*   _loc uses device random value                  mc_global = 1     */
*   _fet uses vt_mc fet type global random value   mc_global = 2     */
*   _loc uses device random values                 mc_global = 0     */
*    uses cor_acv global values                    mc_sw  = 0        */
*  mc_vt    distribution by fet type (see devxfet.model)*/
*  cor_acv  corner parameter for all acv (see process.model) */
*  match_rss rss parameter for local dist and mc_vt from Tech_Data_Xfet  */
.param mm_dop_rss = 0.40
.param vth0mis1 = -0.00114
.param vth0mis2 = 0.00863
.param vth0mis3 = -0.0122
.param vth0mis4 = 0.00731
.param vth0mis5 = 0.00
.param pvtw1 = -0.192
.param pvtw2 = -0.456
.param pvtw3 = -0.476
.param pvtw4 = -0.4
.param pvtw5 = -0.0
.param pvtl1 = -0.192
.param pvtl2 = -0.12
.param pvtl3 = -0.396
.param pvtl4 = -0.346
.param pvtl5 = -0.0
.param u0mis1 = 0.00147
.param u0mis2 = -0.0061
.param u0mis3 = -0.0078
.param u0mis4 = 0.00277
.param u0mis5 = 0.0
.param pu0w1 = -0.205
.param pu0w2 = -0.5
.param pu0l1 = -0.205
.param pu0l2 = -0.6
.param pu03 = -0.667
.param vsatmis1 = 0.0
.param vsatmis2 = 0.0
.param vsatmis3 = 0.0
.param vsatmis4 = -0.00277
.param vsatmis5 = 0.00784
.param rdswmis1 = 0.0
.param rdswmis2 = 0.0061
.param rdswmis3 = 0.0078
.param rdswmis4 = 0.0
.param rdswmis5 = 0.0
.param prdsww1 = 0.0
.param prdsww2 = -0.417
.param prdswl1 = 0.0
.param prdswl2 = -0.42
.param prdsw3 = -0.667
.param k1mis1 = 0.01
.param k1mis2 = 0.0
.param k1mis3 = 0.0
.param k1mis4 = 0.00147
.param k1mis5 = 0.0
.param pk11 = -0.3
.param pk12 = -0.151
.param pk13 = -0.245
.param pk14 = -0.4174
.param pk15 = 0.0
.param nfactormis1 = 0.0
.param nfactormis2 = -0.0194
.param nfactormis3 = -0.00633
.param nfactormis4 = 0.0
.param nfactormis5 = 0.0
.param pnf1 = -0.3
.param pnf2 = -0.342
.param pnf3 = -0.3
.param pnf4 = 0.0
.param pnf5 = 0.0
.param eta0mis1 = -0.00165
.param eta0mis2 = -0.0215
.param eta0mis3 = 0.0
.param eta0mis4 = 0.0525
.param eta0mis5 = 0.0
.param peta1 = -0.446
.param peta2 = -0.455
.param peta3 = 0.0
.param peta4 = -0.0779
.param dvtp1mis4 = 0.272
.param dvtp1mis5 = 0.0
.param pdvtp14 = 0.082
.param pdvtp15 = 0.0
.param pdiblc2mis4 = 0.0
.param pdiblc2mis5 = -0.28
.param ppdiblc24 = 0.0
.param ppdiblc25 = -0.2

.param       mmdist12  = '(mc_vt_pfet * 3)   * (mm_dop_rss)'
.param       mmdist13  = '(psw_acv_sign * cnr_acv_uncorrelated) * (mm_dop_rss)'
.param       mm_on = '((fet_dop_mm == 1) && (pdevdops==1))' 
.param       mm_global = '(mc_sw == 0)* mm_on'
.param       mm_fet    = '(mc_global == 2)*(mc_sw == 1)* mm_on'
.param       mm_local  =  '(mc_global != 2)*(mc_sw == 1)* mm_on'

.param vth0_mult = '1.0 + (0.7071*(vth0mis1* pwr(pwpar,pvtw1)*pwr(p_leff,pvtl1)*(mismatchd1))*mm_local+ 0.7071*(vth0mis2* pwr(pwpar,pvtw2)*pwr(p_leff,pvtl2)*(mismatchd2))*mm_local+ 0.7071*(vth0mis3* pwr(pwpar,pvtw3)*pwr(p_leff,pvtl3)*(mismatchd3))*mm_local + 0.7071*(vth0mis4* pwr(pwpar,pvtw4)*pwr(p_leff,pvtl4)*(mismatchd4))*mm_local+ 0.7071*(vth0mis5* pwr(pwpar,pvtw5)*pwr(p_leff,pvtl5)*(mismatchd5)))*mm_local'
.param vth0_add2 = '(1.0 - (0.7071*(sqrt((vth0mis2*vth0mis2) + (vth0mis3*vth0mis3))*pmm_area*mmdist12)))*(mm_fet)'
.param vth0_add3= '(1.0 - (0.7071*(sqrt((vth0mis2*vth0mis2) + (vth0mis3*vth0mis3))*pmm_area*mmdist13)))*(mm_global)'
.param vth0_add_m = '(((vth0_add2+vth0_add3)-1)*vth0_pfet_n)*(mm_fet+mm_global)'

.param u0_mult1 = '1 + (0.7071*(u0mis2* pwr(pwpar,pu0w2)*pwr(p_leff,pu0l2)*(mismatchd2))+ 0.7071*(u0mis3* pwr(pwpar,pu03)*pwr(p_leff,pu03)*(mismatchd3)) + 0.7071*(u0mis4* pmm_area * (mismatchd4))+ 0.7071*(u0mis5* pmm_area * (mismatchd5)))*mm_local'
.param u0_mult2 = '1.0 + (0.7071*(sqrt((u0mis2*u0mis2) + (u0mis3*u0mis3))*pmm_area*mmdist12)*(mm_fet))'
.param u0_mult3 = '1.0 + (0.7071*(sqrt((u0mis2*u0mis2) + (u0mis3*u0mis3))*pmm_area*mmdist13)*(mm_global))'
.param u0_mult= 'u0_mult1*u0_mult2*u0_mult3'

.param vsat_mult = '1/(1.0 + (0.7071*(vsatmis1 * pmm_area * (mismatchd1))+ 0.7071*(vsatmis2 * pmm_area * (mismatchd2))+ 0.7071*(vsatmis3 * pmm_area * (mismatchd3))+ 0.7071*(vsatmis4 * pmm_area * (mismatchd4))+ 0.7071*(vsatmis5 * pmm_area * (mismatchd5)))*mm_local)'

.param rdsw_mult1 = '(1/(1.0 + (mm_local*0.7071*(rdswmis1* pwr(pwpar,prdsww1)*pwr(p_leff,prdswl1)*(mismatchd1))+ 0.7071*(rdswmis2* pwr(pwpar,prdsww2)*pwr(p_leff,prdswl2)*(mismatchd2)) + 0.7071*(rdswmis3* pwr(pwpar,prdsw3)*pwr(p_leff,prdsw3)*(mismatchd3)))*mm_local))'
.param   rdsw_mult2 = '1/(1.0 + (0.7071*(sqrt((rdswmis2*rdswmis2) + (rdswmis3*rdswmis3))*pmm_area*mmdist12)*(mm_fet)))'
.param   rdsw_mult3 = '1/(1.0 + (0.7071*(sqrt((rdswmis2*rdswmis2) + (rdswmis3*rdswmis3))*pmm_area*mmdist13)*(mm_global)))'
.param rdsw_mult= 'rdsw_mult1*rdsw_mult2*rdsw_mult3'

.param k1_mult1 = '1.0 + 0.7071*(k1mis1* pwr(pwpar,pk11)*pwr(p_leff,pk11)*(mismatchd1))*mm_local+ 0.7071*(k1mis2* pwr(pwpar,pk12)*pwr(p_leff,pk12)*(mismatchd2))*mm_local+ 0.7071*(k1mis3* pwr(pwpar,pk13)*pwr(p_leff,pk13)*(mismatchd3))*mm_local+ 0.7071*(k1mis4* pwr(pwpar,pk14)*pwr(p_leff,pk14)*(mismatchd4))*mm_local+ 0.7071*(k1mis5* pwr(pwpar,pk15)*pwr(p_leff,pk15)*(mismatchd5))*mm_local'
.param  k1_mult2 = '1.0 - (0.7071*(sqrt((k1mis1*k1mis1))*pmm_area*mmdist12))*(mm_fet)'
.param  k1_mult3 = '1.0 - (0.7071*(sqrt((k1mis1*k1mis1))*pmm_area*mmdist13))*(mm_global)'
.param k1_mult= 'k1_mult1*k1_mult2*k1_mult3'

.param nfactor_mult = '1.0 + 0.7071*(nfactormis1* pwr(pwpar,pnf1)*pwr(p_leff,pnf1)*(mismatchd1))*mm_local+ 0.7071*(nfactormis2* pwr(pwpar,pnf2)*pwr(p_leff,pnf2)*(mismatchd2))*mm_local+ 0.7071*(nfactormis3* pwr(pwpar,pnf3)*pwr(p_leff,pnf3)*(mismatchd3))*mm_local+ 0.7071*(nfactormis4* pwr(pwpar,pnf4)*pwr(p_leff,pnf4)*(mismatchd4))*mm_local+ 0.7071*(nfactormis5* pwr(pwpar,pnf5)*pwr(p_leff,pnf5)*(mismatchd5))*mm_local'

.param eta0_mult1 = '1.0 + 0.7071*(eta0mis1* pwr(pwpar,peta1)*pwr(p_leff,peta1)*(mismatchd1))*mm_local+ 0.7071*(eta0mis2* pwr(pwpar,peta2)*pwr(p_leff,peta2)*(mismatchd2))*mm_local+ 0.7071*(eta0mis3* pwr(pwpar,peta3)*pwr(p_leff,peta3)*(mismatchd3))*mm_local+ 0.7071*(eta0mis4* pwr(pwpar,peta4)*pwr(p_leff,peta4)*(mismatchd4))*mm_local'
.param eta0_mult2 = '1.0 + (0.7071*(sqrt((eta0mis1*eta0mis1) + (eta0mis2*eta0mis2) + (eta0mis3*eta0mis3))*pmm_area*mmdist12))*(mm_fet)'
.param eta0_mult3 = '1.0 + (0.7071*(sqrt((eta0mis1*eta0mis1) + (eta0mis2*eta0mis2) + (eta0mis3*eta0mis3))*pmm_area*mmdist13))*(mm_global)'
.param eta0_mult='eta0_mult1*eta0_mult2*eta0_mult3'

.param dvtp1_mult = '1.0 + 0.7071*(dvtp1mis4* pwr(pwpar,pdvtp14)*pwr(p_leff,pdvtp14)*(mismatchd4))*mm_local+ 0.7071*(dvtp1mis5* pwr(pwpar,pdvtp15)*pwr(p_leff,pdvtp15)*(mismatchd5))*mm_local'
.param pdiblc2_mult = '1.0 + 0.7071*(pdiblc2mis4* pwr(pwpar,ppdiblc24)*pwr(p_leff,ppdiblc24)*(mismatchd4))*mm_local+ 0.7071*(pdiblc2mis5* pwr(pwpar,ppdiblc25)*pwr(p_leff,ppdiblc25)*(mismatchd5))*mm_local'
.param dvtp1_pfet    = 0.4
.param pdiblc2_pfet  = 1e-10
.param k1_pfet       = 0.310
.param nfactor_pfet  = 1.58373
.param u0_pfet_f    = 'u0_pfet*u0_mult'
.param vsat_pfet_f  = 'vsat_pfet*vsat_mult' 
.param rdsw_pfet_f  = 'rdsw_pfet*rdsw_mult'
.param lvsat_pfet_f   = 'lvsat_pfet*vsat_mult'
.param eta0_pfet_f   = 'eta0_pfet*eta0_mult'
.param leta0_pfet_f   = 'leta0_pfet*eta0_mult'
.param lrdsw_pfet_f   = 'lrdsw_pfet*rdsw_mult'
.param lu0_pfet_f   = 'lu0_pfet*u0_mult'
.param wu0_pfet_f   = 'wu0_pfet*u0_mult'
.param lk1_pfet_f   = 'lk1_pfet*k1_mult'
.param lnfactor_pfet_f   = 'lnfactor_pfet*nfactor_mult'
.param  p_vttotalvar_f  = '(vth0_mult*vth0_pfet)+vth0_add_m-prox_term +p_vta'
.param  dvtp1_pfet_f    = 'dvtp1_mult*dvtp1_pfet'
.param  pdiblc2_pfet_f  = 'pdiblc2_mult*pdiblc2_pfet'
.param  k1_pfet_f       = 'k1_mult*k1_pfet'
.param  nfactor_pfet_f  = 'nfactor_mult*nfactor_pfet'
**********************************************************************************
************RF Parameters **********************
.param
+  xrcrg1_pfet = 12
+  xrcrg2_pfet =1
+  rshg_pfet = 8
+  xgw_pfet =0.
+  xgl_pfet = .01u
*****************************************************
main  1 2 3 4 pfet l=l w=w ad=ad as=as pd=pd ps=ps  nf=nf nrd=nrd nrs=nrs dtemp = dtemp sa=sa_f sb=sb_f sd=sd_f lmlt=lmlt vmlt=vmlt omlt=omlt
*********************************************
*pfet model for nw proximity
*********************************************
.model  pfet  pmos (level=54
+ version=4.30           mobmod=0                 capmod=2                 
+ diomod=1               trnqsmod=0               acnqsmod=0               
+ fnoimod=1              tnoimod=0                rdsmod=0                 
+ permod=1               geomod=0                 igcmod=1                 
+ igbmod=1               tempmod=0                binunit=1                
+ paramchk=0             toxm=2.38e-09            epsrox=3.90000e+00       
+ cdsc=9.07068e-05       cdscb=0                  cdscd=0.00111342         
+ cit=0                  xj=3e-09                 at=0                     
+ a0=2.14204             ags=0.813559             a1=0.00186087            
+ a2=0.93                keta=0                   ndep=6.79001e+17         
+ nsd=1.00000e+20        ngate=1.0e+20            kt1=-0.185082            
+ kt1l=-3.28327e-09      kt2=-0.0316699           k2=0                     
+ k3b=3.19332            w0=3.78246e-08           dvtp0=4.65696e-07        
+ lpe0=1.0e-9            lpeb=4.22375e-08         dvt1=0.772322            
+ dvt2=-0.0382631        dvt1w=1.29963e+06        dvt2w=-0.38341           
+ drout=0.317167         dsub=0.843644            ua=3.08915e-10           
+ ua1=1e-10              ub=2e-24                 ub1=1.0948e-19           
+ uc=-1.95549e-10        eu=1.67                  ute=-0.8                 
+ voff=-0.136324         minv=3.60048             voffl=-3.16857e-07       
+ vfb=-1                 delta=0.00665612         rdswmin=0                
+ rdwmin=1800            rswmin=1800              prwg=-0.0897756          
+ prwb=0.396868          prt=0                    etab=-0.3                
+ pclm=0.172916          pdiblc1=0.0512379        pdiblcb=-0.129864        
+ fprout=563.712         pdits=1.28802            pditsd=0.59353           
+ pditsl=1.44422e+07     pscbe1=1.43088e+09       pscbe2=1.13852e-06       
+ pvag=8                 wr=1                     dwg=-9.68967e-09         
+ dwb=0                  b0=2.58758e-06           b1=2.41958e-06           
+ alpha0=2.14000e-08     alpha1=7.00000e-02       beta0=1.20000e+01        
+ agidl=1.50981e-08      bgidl=2.91039e+09        cgidl=0.298824           
+ egidl=0.70657          aigc=0.00624             bigc=0.00108622          
+ cigc=0.0266339         aigsd=0.00522957         bigsd=0.00125121         
+ cigsd=0.256217         aigbacc=0.0106209        bigbacc=0.00375221       
+ cigbacc=0.247138       nigbacc=4.10724          aigbinv=0.0131937        
+ bigbinv=0.00492461     cigbinv=0.0871904        eigbinv=1.14103          
+ nigbinv=35.6682        nigc=1.40428             ntox=1                   
+ pigcd=3                poxedge=1.06239          toxref=2.38e-09          
+ ijthsfwd=1.00000e-01   ijthsrev=1.00000e-01     xjbvs=1                  
+ bvs=1.08000e+01        tnoia=1.50000e+00        tnoib=3.50000e+00        
+ ntnoi=1                ckappas=1.90             cf=0                     
+ clc=1.00000e-07        cle=6.00000e-01          xw=0                     
+ xl=0                   dlcig=1.855e-08          noff=2.600e+00           
+ voffcv=-0.08           acde=6.10000e-01         moin=1.40000e+01         
+ lcdsc=2.35302e-05      lcdscb=-4.38109e-06      lcdscd=3.98962e-05       
+ lat=5496.22            lketa=-0.00217537        lk2=-0.000898389         
+ lk3=0.143981           lk3b=-0.250562           llpe0=2.05e-09           
+ llpeb=4.21632e-09      ldvt2=-0.00052816        lua=-4.58353e-12         
+ lua1=-3.32019e-12      lub1=-3.19266e-20        luc=8.22361e-12          
+ luc1=-2.67704e-12      lminv=-0.16287           ldelta=0.00269806        
+ lprt=0.651948          letab=0.0058003          wminv=0.0321416          
+ wdelta=0.000549118     tnom=2.50000e+01         cgbo=0                   
+ xpart=0                njs=1.000e+00            xtis=3.00000e+00         
+ llc=0                  lw=0                     lwc=0                    
+ lwlc=0                 lmin=8.00000e-08         lmax=1.00000e-05         
+ wlc=0                  wln=1                    ww=0                     
+ wwc=0                  wwn=1                    wwl=0                    
+ wwlc=0                 wmin=1.20000e-07         wmax=1.00000e-03         
+ saref=5e-6             sbref=5e-6               wlod=0.9e-7              
+ ku0=1.01e-8            lku0=0.0                 wku0=-1.73e-7             
+ pku0=0                 tku0=0                   kvsat=0                  
+ kvth0=-2.3e-9          lkvth0=0.0               wkvth0=0                 
+ llodku0=4.0            wlodku0=1.0              llodvth=1.0              
+ wlodvth=0              stk2=0                   steta0=0                 
+ em=4.10e+07            ef=1.062                 af=1                     


***************** statistical parameters ***************


+ rbodymod=rbodymod         rgatemod=rgatemod         ngcon=ngcon              
+ xrcrg1=xrcrg1_pfet        xrcrg2=xrcrg2_pfet        rshg=rshg_pfet           
+ xgw = xgw_pfet            xgl = xgl_pfet            nfactor=nfactor_pfet_f   
+ toxe='omlt*toxe_pfet'     toxp='omlt*toxp_pfet'     rdsw=rdsw_pfet_f         
+ u0=u0_pfet_f              vth0='0.3*(vmlt-1)+p_vttotalvar_f'       rsh=rsh_pfet             
+ lwn=lwn_pfet              lwl=lwl_pfet              wl=wl_pfet               
+ uc1=uc1_pfet              k3=k3_pfet                dvt0=dvt0_pfet           
+ dvt0w=dvt0w_pfet          vsat=vsat_pfet_f          wint=wint_pfet           
+ ll=ll_pfet                lln=lln_pfet              k1=k1_pfet_f             
+ lint='lint_pfet+(1-lmlt)*3e-8'            dvtp1=dvtp1_pfet_f        pdiblc2 = pdiblc2_pfet_f 
+ lvsat=lvsat_pfet_f        eta0=eta0_pfet_f          leta0=leta0_pfet_f        
+ lrdsw=lrdsw_pfet_f        lu0=lu0_pfet_f            wu0=wu0_pfet_f            
+ lk1=lk1_pfet_f            lnfactor=lnfactor_pfet_f  
*********************************************************
**capacitance-voltage (cv) parameters
**********************************************************
+ cgsl=cgsl_pfet            cgdl=cgdl_pfet            cgso=cgso_pfet           
+ cgdo=cgdo_pfet            dlc=dlc_pfet              dwc=dwc_pfet             
+ dwj=dwj_pfet             
**********************************************************
**flicker-noise parameters
**********************************************************
+ noia=noia_pfet            noib=noib_pfet            noic=noic_pfet           
+ kf=kf_pfet               
*********************************************************
**Source/Drain parasitic junction diodes
**********************************************************
+ tcj=tcj_pfet              tcjsw=tcjsw_pfet          tpb=tpb_pfet             
+ tpbsw=tpbsw_pfet          tpbswg=tpbswg_pfet        jss=jss_pfet             
+ jsws=jsws_pfet            jswgs=jswgs_pfet          pbs=pbs_pfet             
+ pbsws=pbsws_pfet          pbswgs=pbswgs_pfet        mjs=mjs_pfet             
+ mjsws=mjsws_pfet          cjs=cjs_pfet              cjd=cjd_pfet             
+ cjsws=cjsws_pfet          cjswd=cjswd_pfet          mjswgs=mjswgs_pfet       
+ cjswgs=cjswgs_pfet        cjswgd=cjswgd_pfet       
+ tcjswg=tcjswg_pfet )   
.ends pfet
