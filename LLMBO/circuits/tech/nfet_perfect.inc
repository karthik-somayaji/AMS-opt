* Path, Component, Release: cmos9sf/rel/HSPICE/models/nfet.inc, 9sf_models, mod_cmos9sf
* CMVC Revision: 1.8 06/02/17 13:37:56
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
.subckt nfet 1 2 3 4 
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
.param lp = 'l-(2*lint_nfet_n)'
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
.param   xx_lint_4='((mc_global==2)*(-x_lint_nfet)*(mc_sw)+(mc_global!=2))'
.param   xc_lint_fet='((xc_lint_1+xc_lint_2)*xx_lint_3)*xx_lint_4'
.param   lintchp2    = 'lint_tol_nfet*x_lint_nfet + lint_tol_nfet/2*x_lint_nfet1 '
.param   lintchp3    = '(abs(lchiptol/(lint_tol_nfet*2)))*(len_dep == 1)+ (len_dep != 1)'
.param   lintchpf  = 'lintchp2*lintchp3'

******/*final lint   */
.param   lint_nfet ='lint_nfet_n+0.5*xc_lint_fet*fet_geo_mm*pdevlgeos+  lintchpf + lint_nfet_add'
.param   xc_wint_1='(((acv_uncorrelated*(mc_sw==0))+(acwv_ran*(mc_fet_mm_f))+(mc_global==2))*acwv_fet_ran)'
.param   xx_wint_2='((mc_global==2)*(-x_wint_nfet)*(mc_sw)+(mc_global!=2))'
.param   xc_wint_nfet='xc_wint_1*xx_wint_2'
.param   wint_nfet='wint_nfet_n-0.5*xc_wint_nfet*fet_geo_mm*pdevwgeos+(wint_nfet_n-wint_nfet_b)*x_wint_nfet + wint_nfet_add'
.param      dlc_nfet   = 'dlcslope_nfet*lint_nfet + ladj_nfet'
.param      dwc_nfet   = 'wint_nfet + wadj_nfet'
.param      dwj_nfet   = 'wint_nfet + wjadj_nfet'
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
.param     proxpre_tw = 0.008
+          proxlen_tw = 0.5e-06
+          proxminlen_tw = 0.45e-06
+          proxpre_mid = '(0.016 + proxpre_tw)/2'
+          proxlen_mid = '(0.5e-06 + proxlen_tw)/2'
+          proxminlen_mid = '(0.45e-06 + proxminlen_tw)/2'
+          proxpre= '(0.016*(ptwell==0)) +(proxpre_tw * (ptwell==1)) + (proxpre_mid * (ptwell == 2))'
+          proxlen= '(0.5e-06*(ptwell==0)) +(proxlen_tw * (ptwell==1)) + (proxlen_mid * (ptwell == 2))'
+          proxminlen= '(0.45e-06*(ptwell==0)) +(proxminlen_tw * (ptwell==1)) + (proxminlen_mid * (ptwell == 2))'
+          rxwparm1= -0.22
+          rxwparm2= 0.35e-06
+          rxwparm3 = -3.0
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
+          p_dlnom = '2*lint_nfet'
+          lvsat_nfet = 327.061 
+          wvsat_nfet = 441.64 
+          eta0_nfet = 0.634929 
+          lrdsw_nfet = 6.60445 
+          lu0_nfet = 7.46732 
+          wu0_nfet = -2.8767 
+          pvth0_nfet = -0.0005 
+          wnfactor_nfet = 6.44579e-05 
+          lpdiblc2_nfet = -7.35611e-07 
+          wpdiblc2_nfet = 0 
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
.param mm_dop_rss = 0.86
.param vth0mis1 = -0.00057
.param vth0mis2 = 0.00624
.param vth0mis3 = -0.0022
.param vth0mis4 = 0.00731
.param vth0mis5 = 0.00
.param pvtw1 = -0.417
.param pvtw2 = -0.42
.param pvtw3 = -0.737
.param pvtw4 = -0.4
.param pvtw5 = -0.0
.param pvtl1 = -0.839
.param pvtl2 = -0.37
.param pvtl3 = -0.737
.param pvtl4 = -0.45
.param pvtl5 = -0.0
.param u0mis1 = -0.000167
.param u0mis2 = -0.00691
.param u0mis3 = 0.00691
.param u0mis4 = 0.00277
.param u0mis5 = 0.0
.param pu0w1 = -0.271
.param pu0w2 = -0.392
.param pu0l1 = -1.548
.param pu0l2 = -0.392
.param pu03 = -0.392
.param vsatmis1 = 0.0
.param vsatmis2 = 0.0
.param vsatmis3 = 0.0
.param vsatmis4 = -0.00277
.param vsatmis5 = 0.00784
.param rdswmis1 = 0.0
.param rdswmis2 = 0.00691
.param rdswmis3 = -0.00691
.param rdswmis4 = 0.0
.param rdswmis5 = 0.0
.param prdsww1 = 0.0
.param prdsww2 = -0.392
.param prdswl1 = 0.0
.param prdswl2 = -0.392
.param prdsw3 = -0.392
.param k1mis1 = 0.00553
.param k1mis2 = -0.00485
.param k1mis3 = -0.00295
.param k1mis4 = 0.00147
.param k1mis5 = 0.0
.param pk11 = -0.392
.param pk12 = -0.151
.param pk13 = -0.245
.param pk14 = -0.4174
.param pk15 = 0.0
.param nfactormis1 = 0.00297
.param nfactormis2 = -0.04
.param nfactormis3 = -0.04854
.param nfactormis4 = 0.0
.param nfactormis5 = 0.0
.param pnf1 = -0.3
.param pnf2 = -0.262
.param pnf3 = -0.255
.param pnf4 = 0.0
.param pnf5 = 0.0
.param eta0mis1 = 0.0
.param eta0mis2 = -0.0215
.param eta0mis3 = 0.0
.param eta0mis4 = 0.0657
.param eta0mis5 = 0.0
.param peta1 = 0.0
.param peta2 = -0.455
.param peta3 = 0.0
.param peta4 = -0.0779
.param dvtp1mis4 = 0.389
.param dvtp1mis5 = 0.0
.param pdvtp14 = 0.082
.param pdvtp15 = 0.0
.param pdiblc2mis4 = 0.0
.param pdiblc2mis5 = -0.28
.param ppdiblc24 = 0.0
.param ppdiblc25 = -0.2

.param       mmdist12  = '(mc_vt_nfet * 3)   * (mm_dop_rss)'
.param       mmdist13  = '(psw_acv_sign * cnr_acv_uncorrelated) * (mm_dop_rss)'
.param       mm_on = '((fet_dop_mm == 1) && (pdevdops==1))' 
.param       mm_global = '(mc_sw == 0)* mm_on'
.param       mm_fet    = '(mc_global == 2)*(mc_sw == 1)* mm_on'
.param       mm_local  =  '(mc_global != 2)*(mc_sw == 1)* mm_on'

.param vth0_mult = '1.0 + (0.7071*(vth0mis1* pwr(pwpar,pvtw1)*pwr(p_leff,pvtl1)*(mismatchd1))*mm_local+ 0.7071*(vth0mis2* pwr(pwpar,pvtw2)*pwr(p_leff,pvtl2)*(mismatchd2))*mm_local+ 0.7071*(vth0mis3* pwr(pwpar,pvtw3)*pwr(p_leff,pvtl3)*(mismatchd3))*mm_local + 0.7071*(vth0mis4* pwr(pwpar,pvtw4)*pwr(p_leff,pvtl4)*(mismatchd4))*mm_local+ 0.7071*(vth0mis5* pwr(pwpar,pvtw5)*pwr(p_leff,pvtl5)*(mismatchd5)))*mm_local'
.param vth0_add2 = '(1.0 - (0.7071*(sqrt((vth0mis2*vth0mis2) + (vth0mis3*vth0mis3))*pmm_area*mmdist12)))*(mm_fet)'
.param vth0_add3= '(1.0 - (0.7071*(sqrt((vth0mis2*vth0mis2) + (vth0mis3*vth0mis3))*pmm_area*mmdist13)))*(mm_global)'
.param vth0_add_m = '(((vth0_add2+vth0_add3)-1)*vth0_nfet_n)*(mm_fet+mm_global)'

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
.param dvtp1_nfet    = 0.587259
.param pdiblc2_nfet  = 0.000366969
.param k1_nfet       = 0.301986
.param nfactor_nfet  = 1.12582
.param u0_nfet_f    = 'u0_nfet*u0_mult'
.param vsat_nfet_f  = 'vsat_nfet*vsat_mult' 
.param rdsw_nfet_f  = 'rdsw_nfet*rdsw_mult'
.param lvsat_nfet_f   = 'lvsat_nfet*vsat_mult'
.param wvsat_nfet_f   = 'wvsat_nfet*vsat_mult'
.param eta0_nfet_f   = 'eta0_nfet*eta0_mult'
.param lrdsw_nfet_f   = 'lrdsw_nfet*rdsw_mult'
.param lu0_nfet_f   = 'lu0_nfet*u0_mult'
.param wu0_nfet_f   = 'wu0_nfet*u0_mult'
.param pvth0_nfet_f   = 'pvth0_nfet*vth0_mult'
.param wnfactor_nfet_f   = 'wnfactor_nfet*nfactor_mult'
.param lpdiblc2_nfet_f   = 'lpdiblc2_nfet*pdiblc2_mult'
.param wpdiblc2_nfet_f   = 'wpdiblc2_nfet*pdiblc2_mult'
.param  p_vttotalvar_f  = '(vth0_mult*vth0_nfet)+vth0_add_m+prox_term +p_vta'
.param  dvtp1_nfet_f    = 'dvtp1_mult*dvtp1_nfet'
.param  pdiblc2_nfet_f  = 'pdiblc2_mult*pdiblc2_nfet'
.param  k1_nfet_f       = 'k1_mult*k1_nfet'
.param  nfactor_nfet_f  = 'nfactor_mult*nfactor_nfet'
**********************************************************************************
************RF Parameters **********************
.param
+  xrcrg1_nfet = 12
+  xrcrg2_nfet =1
+  rshg_nfet = 8
+  xgw_nfet =0.
+  xgl_nfet = .01u
*****************************************************
main  1 2 3 4 nfet l=l w=w ad=ad as=as pd=pd ps=ps  nf=nf nrd=nrd nrs=nrs dtemp = dtemp sa=sa_f sb=sb_f sd=sd_f
*********************************************
*nfet model for nw proximity
*********************************************
.model  nfet  nmos (level=54
+ version=4.30           mobmod=0                 capmod=2                 
+ diomod=1               trnqsmod=0               acnqsmod=0               
+ fnoimod=1              tnoimod=0                rdsmod=0                 
+ permod=1               geomod=0                 igcmod=1                 
+ igbmod=1               tempmod=0                binunit=1                
+ paramchk=0             toxm=2.20e-09            epsrox=3.90000e+00       
+ cdsc=0.00096189        cdscb=-0.002             cdscd=0                  
+ cit=0                  xj=5.35217e-09           at=0                     
+ a0=2.22519             ags=0.862388             a1=0.0                   
+ a2=1.0                 keta=-0.00583762         ndep=7.46402e+17         
+ nsd=1.00000e+20        ngate=1.00e+20           kt1=-0.136848            
+ kt1l=-1.10145e-09      kt2=-0.0182137           k2=0                     
+ k3b=1.4698             w0=0                     dvtp0=1.94386e-07        
+ lpe0=2.46053e-08       lpeb=3.10518e-08         dvt1=1.06851             
+ dvt2=0                 dvt1w=6.8002e+06         dvt2w=0.30               
+ drout=0.0685201        dsub=0.871629            ua=1e-14                 
+ ua1=2.08075e-09        ub=2e-26                 ub1=-1.46178e-18         
+ uc=-7.98339e-11        eu=1.67000e+00           ute=-0.7                 
+ voff=-0.189474         minv=3.56963             voffl=-3.98262e-06       
+ vfb=-1                 delta=0.00770419         rdswmin=43.5651          
+ rdwmin=1800            rswmin=1800              prwg=12.0468             
+ prwb=0.407894          prt=0                    etab=-0.144589           
+ pclm=0.871619          pdiblc1=0.00135946       pdiblcb=-0.995518        
+ fprout=97.202          pdits=0.106565           pditsd=-0.0557934        
+ pditsl=5.70551e+06     pscbe1=3.0804e+10        pscbe2=0.000724626       
+ pvag=1.67198           wr=1                     dwg=-4.7126e-09          
+ dwb=-4.50959e-11       b0=1.21449e-07           b1=1e-07                 
+ alpha0=3.00000e-11     alpha1=0                 beta0=3.80000e+00        
+ agidl=2.1956e-08       bgidl=3.73039e+09        cgidl=0.858994           
+ egidl=0.576585         aigc=0.00965             bigc=0.00284             
+ cigc=0.228             aigsd=0.00701239         bigsd=0.00131004         
+ cigsd=0.106974         aigbacc=0.0119339        bigbacc=0.00371637       
+ cigbacc=0.24797        nigbacc=4.50572          aigbinv=0.013            
+ bigbinv=0.004          cigbinv=0.10             eigbinv=0.2              
+ nigbinv=0.646975       nigc=2.801               ntox=1                   
+ pigcd=5                poxedge=1.35766          toxref=2.20e-09          
+ ijthsfwd=1.00000e-01   ijthsrev=1.00000e-01     xjbvs=1                  
+ bvs=1.20000e+01        tnoia=1.50000e+00        tnoib=3.50000e+00        
+ ntnoi=1                ckappas=0.846            cf=0                     
+ vfbcv=-1               clc=1.00000e-07          cle=6.00000e-01          
+ xw=0                   xl=0                     dlcig=1.87291e-08        
+ noff=2.10000e+00       voffcv=-3.00000e-02      acde=1.2                 
+ moin=10.00000e+00      lcdsc=-1.55315e-05       lcdscd=1.05937e-05       
+ lat=3461.21            lua=1.65592e-11          lua1=-3.1599e-11         
+ lub=3.8e-20            lub1=-1.21064e-20        luc=3.55848e-11          
+ luc1=-1.7572e-11       lminv=-0.113539          ldelta=0.00180503        
+ lprwg=0.146234         lprwb=-0.0105738         lprt=2.24774             
+ lpdiblc1=-3.39175e-06  lpdiblcb=-0.000170309    lpscbe1=-5.96444e+08     
+ lpvag=0.506122         wdelta=0.0005            tnom=2.50000e+01         
+ cgbo=0                 xpart=0                  njs=1.05000e+00          
+ njd=1.05000e+00        xtis=3.00000e+00         llc=0                    
+ lw=0                   lwc=0                    lwlc=0                   
+ lmin=0.08e-06          lmax=1.00000e-05         wlc=0                    
+ wln=1                  ww=0                     wwc=0                    
+ wwn=1                  wwl=0                    wwlc=0                   
+ wmin=0.120e-06         wmax=1                   saref=5e-6               
+ sbref=5e-6             wlod=0.0                 ku0=-3.0e-8              
+ lku0=3.0e-8            wku0=2.0e-26             pku0=0                   
+ tku0=0                 kvsat=0                  kvth0=2.8e-9             
+ lkvth0=2.0e-29         wkvth0=0                 llodku0=1                
+ wlodku0=4.0            llodvth=4.0              wlodvth=0                
+ stk2=0                 steta0=0                 em=4.10e+07              
+ ef=0.957               af=1                     


**************** statistical parameters ******************


+ rbodymod=rbodymod         rgatemod=rgatemod         ngcon=ngcon              
+ xrcrg1=xrcrg1_nfet        xrcrg2=xrcrg2_nfet        rshg=rshg_nfet           
+ xgw = xgw_nfet            xgl = xgl_nfet            nfactor=nfactor_nfet_f   
+ toxe=toxe_nfet            toxp=toxp_nfet            rdsw=rdsw_nfet_f         
+ u0=u0_nfet_f              vth0=p_vttotalvar_f       rsh=rsh_nfet             
+ lwn=lwn_nfet              lwl=lwl_nfet              wl=wl_nfet               
+ uc1=uc1_nfet              k3=k3_nfet                dvt0=dvt0_nfet           
+ dvt0w=dvt0w_nfet          vsat=vsat_nfet_f          wint=wint_nfet           
+ ll=ll_nfet                lln=lln_nfet              k1=k1_nfet_f             
+ lint=lint_nfet            dvtp1=dvtp1_nfet_f        pdiblc2 = pdiblc2_nfet_f 
+ lvsat=lvsat_nfet_f        wvsat=wvsat_nfet_f        eta0=eta0_nfet_f          
+ lrdsw=lrdsw_nfet_f        lu0=lu0_nfet_f            wu0=wu0_nfet_f            
+ pvth0=pvth0_nfet_f        wnfactor=wnfactor_nfet_f  lpdiblc2=lpdiblc2_nfet_f  
+ wpdiblc2=wpdiblc2_nfet_f  
*********************************************************
**capacitance-voltage (cv) parameters
**********************************************************
+ cgsl=cgsl_nfet            cgdl=cgdl_nfet            cgso=cgso_nfet           
+ cgdo=cgdo_nfet            dlc=dlc_nfet              dwc=dwc_nfet             
+ dwj=dwj_nfet             
**********************************************************
**flicker-noise parameters
**********************************************************
+ noia=noia_nfet            noib=noib_nfet            noic=noic_nfet           
+ kf=kf_nfet               
*********************************************************
**Source/Drain parasitic junction diodes
**********************************************************
+ tcj=tcj_nfet              tcjsw=tcjsw_nfet          tpb=tpb_nfet             
+ tpbsw=tpbsw_nfet          tpbswg=tpbswg_nfet        jss=jss_nfet             
+ jsws=jsws_nfet            jswgs=jswgs_nfet          pbs=pbs_nfet             
+ pbsws=pbsws_nfet          pbswgs=pbswgs_nfet        mjs=mjs_nfet             
+ mjsws=mjsws_nfet          cjs=cjs_nfet              cjd=cjd_nfet             
+ cjsws=cjsws_nfet          cjswd=cjswd_nfet          mjswgs=mjswgs_nfet       
+ cjswgs=cjswgs_nfet        cjswgd=cjswgd_nfet       
+ tcjswg=tcjswg_nfet )   
.ends nfet
