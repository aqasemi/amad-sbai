I) Introduction
###############

The example script (“customClock_script.R”) runs through the steps of generating and testing a "custom" clock, based on data from NHANES IV. The script will run through the steps to use the 1999/2000 recruitment wave of NHANES IV to build a custom clock before testing the clock in the separate 2001/2002 recruitment wave. The clock is constructed using dimensionality reduction by PCA and a regularized COX proportional hazard model. The entire process is based on a user-defined set of features and several user modifiable parameters. Features are selected for inclusion by editing the codebook file ("codebook_custom_linAge.csv"). Parameters controlling different aspects of the workflow can also be set in a parameter file (“paraInit.csv”). 


Program flow:
###########

   1)  Data is read (“mergedData_PA.csv”) and features to be used are selected (based on codebook)
   2)  Missing values are removed
   3)  Raw feature values are converted (column-wise) into Z-scores
   4)  Data is split into training (1999/2000 recruitment wave) and testing set (2001/2002 recruitment wave)  
   5)  Training data is converted into PCA coordinates by SVD (using only training set)
   6)  Testing data is converted into the same PC coordinate system by projection
   7)  Dimensionality reduction by dropping any PC coordinates explaining less of the variance than user-set cutoff
   8)  Fit separate male/female COX proportional hazard models using chronological age and remaining PCs as covariates  
   9)  Clocks convert COX proportional hazards  (mortality risk) into biological age deltas
   10) Apply resulting clock to subjects of testing cohort (2001/2002 NHANES recruitment wave)
   11) Test prediction by Receiver operating characteristic (ROC) analysis using actual mortality (linkage)
   12) Apply stochastic noise to original input data of testing data, then re-calculate clock to evaluate impact



II) Workflow  
###########

   To generate a custom clock, only 3 steps are required from the user:
   ---------------------------------------------------------------------------------
   A) Edit codebook file ("codebook_custom_linAge.csv") to select features that will be used (see below)

   B) (Optional): Edit parameter file ("paraInit.csv") to change any of the control parameters

   C) From within R, run script ("customClock_script.R") by typing "source("customClock_script.R")".

   NOTE: The script will assume that both the parameter file (“paraInit.csv”) and the NHANES data file   
         ("mergedData_PA.csv") are present in the same working directory.  


The script will generate log and statistical output to stdout in R and also return three graphical outputs in the form of pdfs. 

These graphical objects created are:

1) Scree plot ("scree.pdf"), showing the fraction of variance explained by each PC together with
    a representation of the dimensionalty reduction performed. 
   
2) ROC analysis figure ("ROC.pdf"), comparing chronological age, PhenoAge and the custom clock
    in predicting survival in the testing data (NHANES 2001/2002 recruitment wave). 
    This is similar to the ROC curve shown in Fig. 5c of the manuscript.
 
3) Stacked histogram plot ("histCompNoise.pdf") comparing the impact on PhenoAge and the
    custom clock when stochastic noise (at the level determined by "errorLevel" in parameter file) is added 
    to the raw input data of the testing data (same as in Fig. 2c of the manuscript).  




III) User editable files 
#######################

   1)  Codebook file: codebook_custom_linAge.csv
       -----------------------------------------
The codebook contains information on all parameters (features) included in the datafile (mergedData_PA.csv). These parameters that are flagged with a "1" in the "Data" column will be included in the clock, unless they contain more NA values than specified in the paraInit.csv. If the fraction of NAs is larger than specified by NAcutOffpercent in paraInit.csv, the parameter is dropped. If not, it is used. Parameters that have the "ForceInc" flag set to "1" are included regardless and subjects (rows) with NA value in these parameters are dropped instead. To ensure that a parameter is used, both the "Data" and "ForceInc" need to be set to "1". Generally, only quantitative numerical values should be selected for inclusion - these are mainly parameters of type "LAB" (Laboratory data) or "E" (Medical examination). 

       Codebook format
       ###############

        Row     Var              Human                          Type    Data     ForceInc
        1       SDDSRVYR         Data release cycle             DEMO    0        0
        2       RIDSTATR         Interview/Examination status   DEMO    0        0
                                               [...]                     
        1142    LBXSKSI          Potassium (mmol/L)             LAB     0        0
        1143    LBXSCLSI         Chloride (mmol/L)              LAB     0        0
        1144    LBDSGBSI         Globulin (g/L)                 LAB     0        0


       Human-readable names for parameters can be viewed in the codebook and these data can also be recalled
       from within R using the customClock_script.R script by calling the varNH2Hum function with the
       NHANES Var name as argument.

         Example: > varNH2Hum("LBXSKSI") will return the human readable form: "Potassium (mmol/L)"


   2)  Parameter file: paraInit.csv
       ----------------------------
       This file contains parameters that impact aspects of the fitting procedure for the custom clock. There are currently 6 such parameter. They are explained in the parameter file. 
       

          Parameter file format
          #################

	  rowNr           pName             pValue    Explanation
	  1               errorLevel        0.10      Set error level as fraction of para value (normal distributed)
      	  2               NAcutOffpercent   0.09      Cut off fraction for NAs
          3               PCcutOffpercent   1.00      Cut off percentage for PC based on variance explained 
          4               lowerAgeLimit     20.00     Lower age bracket of cohort
          5               upperAgeLimit     84.00     Upper age bracket of cohort
          6               derivedFeatFlag   1.00      Set to zero (0) to skip calculating derived features

       Most of these parameters can be changed. However, changing errorLevel will not change error levels
       for PhenoAge because PhenoAge values have been pre-calculated.

       If the derivedFeatFlag is set to "1", then several derived features used by linAge are generated 
       based on other features, provided the original features are included by setting the relevant flags 
       in the codebook.
       
       Features that the script will attempted to created are three clinical scores (fs1Scrore, fs2Scrore, fs3Scrore), 
       urine albumin-to-creatinine (A:Cr) ratio and low-density lipoprotein (LDL) cholesterol. 
       Derived features will be used to fit the clock, if generated.  
