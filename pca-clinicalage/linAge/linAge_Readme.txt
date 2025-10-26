This is an xls spreadsheet implementation showing the way linAge is calculated from the parameters file.
The example is for a male subject (SEQN = 10). Column explanation:

Columns:
#######
NHANES IV     : Variable name for input data from NHANES IV and derived variables.
median        : median for variables for subjects of the same sex that are part of the PCAge healthy aging cluster
MAD           : MAD for variables for subjects of the same sex that are part of the PCAge healthy aging cluster
beta          : Linear weight of this parameter in linAge (combining PC rotations and weight in COX model)
Example data  : Example input data from specific subject (see below)
genZ_scores   : Generalized z-scores for each parameter: (value-median)/MAD
beta_i*Zpara_i: z-scores * weights

Calculation:
###########
sum(beta_i*Zpara_i) : Sum over beta_i*Zpara_i column (sum over contributions from all parameters)
C1_male             : C1 parameter for males
beta0_male          : weight parameter for chronological age

Model equation:
##############

                       deltaBioAge = C1 + beta_0*chronAge+sum(beta_i*Z_para_i)"

NOTE: Missing values
####################
Missing values can be imputed. Alternatively, the impact of the missing variable on the clock can be set to
zero by setting the value of the missing variable to the respective MAD value. The magnitude of the error
introduced by this is on the order of the value beta_i for the variable. 


Example data:
############
>>> Example subject - male subject with SEQN = 10 with data:

    SEQN   BPXPLS   BPXSAR   BPXDAR   BMXBMI LBDIRNSI LBDTIBSI   LBXPCT 
   10.00    58.00   142.00    95.00    30.94    14.86    52.98    28.00 
LBDFERSI LBDFOLSI LBDB12SI   LBXCOT  LBDTCSI LBDHDLSI LBXWBCSI LBXLYPCT 
   32.00    30.60   281.92     3.00     3.62     1.31     4.60    41.30 
LBXMOPCT LBXNEPCT LBXEOPCT LBXBAPCT LBDLYMNO  LBDMONO  LBDNENO  LBDEONO 
   10.50    43.80     3.40     1.00     1.90     0.50     2.00     0.20 
 LBDBANO LBXRBCSI   LBXHGB   LBXHCT LBXMCVSI LBXMCHSI    LBXMC   LBXRDW 
    0.00     5.00    15.40    46.20    92.30    30.90    33.50    13.70 
LBXPLTSI  LBXMPSI   LBXCRP  LBDFBSI    LBXGH    SSBNP LBDSALSI LBXSATSI 
  167.00     9.40    -2.12     3.13     5.50     3.67    42.00    20.00 
LBXSASSI LBXSAPSI LBDSBUSI LBDSCASI LBXSC3SI LBXSGTSI LBDSGLSI LBXSLDSI 
   26.00    86.00     4.60     2.35    21.00    25.00     4.83   201.00 
LBDSPHSI LBDSTBSI LBDSTPSI LBDSTRSI LBDSUASI LBDSCRSI LBXSNASI  LBXSKSI 
    0.97     6.80    76.00     0.47   356.90    79.60   140.90     4.28 
LBXSCLSI LBDSGBSI  
  106.30    34.00
