## This script is a stand-alone linAge calculator.
## It expects to find the following times in the current working directory:
##
## *) A data matrix in NHANES format (using NHANES IV variable names)
## *) A qData matrix of the relevant NHANES questionnaire data
## *) A parameter file giving the linear weights and variable names for linAge
##
## The data matrix should include the following variables, needed to run the model:
## (Variable names are from NHANES IV 99/00)
##
## [1]  "SEQN"     "BPXPLS"   "BPXSAR"   "BPXDAR"   "BMXBMI"  
## [7]  "LBDIRNSI" "LBDTIBSI" "LBXPCT"   "LBDFERSI" "LBDFOLSI" "LBDB12SI"
## [13] "LBXCOT"   "LBDTCSI"  "LBDHDLSI" "LBXWBCSI" "LBXLYPCT" "LBXMOPCT"
## [19] "LBXNEPCT" "LBXEOPCT" "LBXBAPCT" "LBDLYMNO" "LBDMONO"  "LBDNENO" 
## [25] "LBDEONO"  "LBDBANO"  "LBXRBCSI" "LBXHGB"   "LBXHCT"   "LBXMCVSI"
## [31] "LBXMCHSI" "LBXMC"    "LBXRDW"   "LBXPLTSI" "LBXMPSI"  "LBXCRP"  
## [37] "LBDFBSI"  "LBXGH"    "SSBNP"    "LBDSALSI" "LBXSATSI" "LBXSASSI"
## [43] "LBXSAPSI" "LBDSBUSI" "LBDSCASI" "LBXSC3SI" "LBXSGTSI" "LBDSGLSI"
## [49] "LBXSLDSI" "LBDSPHSI" "LBDSTBSI" "LBDSTPSI" "LBDSTRSI" "LBDSUASI"
## [55] "LBDSCRSI" "LBXSNASI" "LBXSKSI"  "LBXSCLSI" "LBDSGBSI" "URXUCRSI"
## [61] "URXUMASI" 
##
## In addition, the following questionnaire variables are required in qDataMatrix
## (Variable names from NHANES IV 99/00 wave)
## BPQ020  DIQ010  HUQ010  HUQ020  HUQ050  HUQ070   KIQ020  MCQ010  MCQ053 
## MCQ160A MCQ160B MCQ160C MCQ160D MCQ160E MCQ160F  MCQ160G MCQ160I 
## MCQ160J MCQ160K MCQ160L MCQ220  OSQ010A OSQ010B  OSQ010C OSQ060  PFQ056 
## HUQ010  HUQ020  HUQ050  HUQ070  HUQ050  RIAGENDR RIDAGEYR

library(survival)
library(ggplot2)

## Workflow:
###########
## 1) Read in dataMatrix (biological parameters)
## 2) Read in qDataMatrix (questionnaire and demographic data)
## 3) Read in linAge model parameters (median,MAD,beta for each relevant par)
## 4) Calculate derived parameters (for linAge: LDLV,crAlbRat,fs1,fs2,fs3)
## 5) Apply appropriate normalization:
##           * log(CRP)
##           * log(BNP)
##           * digitize cotinine values
## 6) Apply linear clock from linAge parameter file - there are separate ones for male and
##    for female subjects

########################################################################################
##    Data cleanup and derived parameters - summary of steps:                         ##
########################################################################################
##    Following the workflow above (as we would for a new matrix) as well
##    we will calculate all the derived parameters and check that these
##    come out identical to the ones we have exported (_Test) versions
##
##    Then we will normalize things according to what we do in the paper
##    (take log of CRP, SSBNP and digitize cotinine values  
##
##    Finally, we will calculate linAge using parameter files (for male and female SEQs)
##
##    Derived parameters:
##                    "fs1Score_Test"    
##                    "fs2Score_Test"
##                    "fs3Score_Test"
##                    "LDLV_Test"        
##                    "crAlbRat_Test"
##
##
##    Normalized parameters:
##                    "LBXCRP_Test" (log)
##                    "SSBNP_Test"  (log) 
##                    "LBXCOT_Test" (Digitize) 
##



## Clean up and get ready 
cat("\n\n\n\n> Cleaning working directory\n\n")
rm(list=ls())


################################################################ 
##                   Function declarations         START >>>  ##
################################################################

digiCot <- function(dataMat) {

    ## Digitize continine to turn into smoking intensity
    ## Most clinics do not routinely measure cotinine - so here we will     
    ## bin cot as follows:
    ## 0  <= cot < 10 are non smokers (0)
    ## 10 >= cot < 100 are light smokers (1)
    ## 100 >= cot < 200 are moderate smokers (2)
    ## anything above 200 is a heavy smoker (3)
    
    cot <- dataMat[,"LBXCOT"]
    dataMat[,"LBXCOT"][cot < 10]<-0
    dataMat[,"LBXCOT"][(cot >= 10) & (cot < 100)]<- 1
    dataMat[,"LBXCOT"][(cot >= 100) & (cot < 200)]<- 2
    dataMat[,"LBXCOT"][(cot >= 200)]<- 3

    ## Now return data matrix with digitized cot values
    return(dataMat) 
}

popPCFIfs1 <- function(qDataMat) {
        ## This will calculate a frailty index / disease and comorbidity index (the FS scale) for each subject
        ## and populate the matrix

        ## NOTE: we will allow NAs here - so check that the variables are all there
        BPQ020 <- qDataMat[,"BPQ020"]
        DIQ010<- qDataMat[,"DIQ010"]
        HUQ010 <- qDataMat[,"HUQ010"]
        HUQ020 <- qDataMat[,"HUQ020"]
        HUQ050  <- qDataMat[,"HUQ050"]
        HUQ070 <- qDataMat[,"HUQ070"]
        KIQ020  <- qDataMat[,"KIQ020"]
        MCQ010 <- qDataMat[,"MCQ010"]
        MCQ053  <- qDataMat[,"MCQ053"]
        MCQ160A <- qDataMat[,"MCQ160A"]
        MCQ160B  <- qDataMat[,"MCQ160B"]
        MCQ160C <- qDataMat[,"MCQ160C"]
        MCQ160D  <- qDataMat[,"MCQ160D"]              
        MCQ160E  <- qDataMat[,"MCQ160E"] 
        MCQ160F <- qDataMat[,"MCQ160F"]
        MCQ160G <- qDataMat[,"MCQ160G"]
        MCQ160I <- qDataMat[,"MCQ160I"]
        MCQ160J <- qDataMat[,"MCQ160J"]
        MCQ160K <- qDataMat[,"MCQ160K"]
        MCQ160L <- qDataMat[,"MCQ160L"]
        MCQ220 <- qDataMat[,"MCQ220"]
        OSQ010A <- qDataMat[,"OSQ010A"]
        OSQ010B <- qDataMat[,"OSQ010B"]
        OSQ010C <- qDataMat[,"OSQ010C"]
        OSQ060 <- qDataMat[,"OSQ060"]
        PFQ056 <- qDataMat[,"PFQ056"]

        ## Give "safe" value to all NAs ... 
        BPQ020[is.na(BPQ020)] <- 2
        DIQ010[is.na(DIQ010)] <- 2
        HUQ010[is.na(HUQ010)] <- 3
        HUQ020[is.na(HUQ020)] <- 3
        HUQ050[is.na(HUQ050)] <- 0
        HUQ070[is.na(HUQ070)] <- 2
        KIQ020[is.na(KIQ020)] <- 2 
        MCQ010[is.na(MCQ010)] <- 2
        MCQ053[is.na(MCQ053)] <- 2
        MCQ160A[is.na(MCQ160A)] <- 2
        MCQ160B[is.na(MCQ160B)]  <- 2
        MCQ160C[is.na(MCQ160C)] <-  2
        MCQ160D[is.na(MCQ160D)]  <- 2
        MCQ160E[is.na(MCQ160E)]  <- 2
        MCQ160F[is.na(MCQ160F)] <-  2
        MCQ160G[is.na(MCQ160G)] <- 2
        MCQ160I[is.na(MCQ160I)] <- 2
        MCQ160J[is.na(MCQ160J)] <- 2
        MCQ160K[is.na(MCQ160K)] <- 2
        MCQ160L[is.na(MCQ160L)] <- 2
        MCQ220[is.na(MCQ220)] <- 2
        OSQ010A[is.na(OSQ010A)] <- 2
        OSQ010B[is.na(OSQ010B)] <- 2
        OSQ010C[is.na(OSQ010C)] <- 2
        OSQ060[is.na(OSQ060)] <- 2
        PFQ056[is.na(PFQ056)] <- 2

        ## Binary yes/no vecoro 
        binVec <- cbind((BPQ020==1),((DIQ010==1) | (DIQ010==3)),(KIQ020==1),(MCQ010==1),(MCQ053==1),(MCQ160A==1),(MCQ160C==1),(MCQ160D==1),(MCQ160E==1),(MCQ160F==1),(MCQ160G==1),(MCQ160I==1),(MCQ160J==1),(MCQ160K==1),(MCQ160L==1),(MCQ220==1),(OSQ010A==1),(OSQ010B==1),(OSQ010C==1),(OSQ060==1),(PFQ056==1),(HUQ070==1))

    sumOverBinVec <- rowSums(binVec)/22 
    return(sumOverBinVec)
}




popPCFIfs2 <- function(qDataMat) {
        
    HUQ010 <- qDataMat[,"HUQ010"]
    HUQ020 <- qDataMat[,"HUQ020"]
    HUQ050  <- qDataMat[,"HUQ050"]
    HUQ070 <- qDataMat[,"HUQ070"]
    HUQ010[is.na(HUQ010)] <- 3
    HUQ020[is.na(HUQ020)] <- 3
    HUQ050[is.na(HUQ050)] <- 0
    HUQ070[is.na(HUQ070)] <- 2

    ## If ill, get score of 2 to 4 - but if getting worse, get x2 or better 1/2 modifier 
    aVec <- ((HUQ010==4)*2+(HUQ010==5)*4)
    dVec <- (1-(HUQ020==1)*0.5+(HUQ020==2))
    fScore <- aVec*dVec
        
    return(fScore)
}



popPCFIfs3 <- function(qDataMat) {
        
    HUQ050  <- qDataMat[,"HUQ050"]
    HUQ050[is.na(HUQ050)] <- 0
    HUQ050[(HUQ050==77)] <- 0
    HUQ050[(HUQ050==99)] <- 0
    return(HUQ050)
}

populateLDL <- function (dataMat,qDataMat) {                                                  
        ## This function will calculate LDL and add it to the dataMatrix
        ## LDL - calculated from:
        ##    Variable: LBDTCSI	Total Cholesterol (mmol/L)
        ##    Variable: LBDHDLSI	HDL (mmol/L)
        ##    Variable: LBDSTRSI	Triglycerides (mmol/L)
        ## Formula:  LDL-C=(TC)–(triglycerides/5)– (HDL-C). 
        ## NOTES: Can be inaccurate if triglycerides are very high (above 150 mg/dL)
        ##        BUT here we will suggest statin / fibrate if TRIGS are high as well, so no concern

        nSubs <- dim(dataMat)[1]
        
        ## Extract all relevant variables from data matrix 
        totCv <- dataMat[,"LBDTCSI"]
        HDLv <- dataMat[,"LBDHDLSI"]
        triGv <- dataMat[,"LBDSTRSI"]
        seqVec <- dataMat[,"SEQN"]
        LDLvec <- rep(0,nSubs)
        
        ## Loop over all subjects and return seqs for anybody qualifying for treatment suggestion 

        for (i in 1:nSubs) {
                    
            totC <- totCv[i]
            HDL <- HDLv[i] 
            TG <- triGv[i] 
            LDL <- 0
            
            ## Check that we do not have any NAs here
            if (!is.na(totC)*!is.na(HDL)*!is.na(TG)) {

                ## Calculate LDL from triglycerides and total cholesterol 
                LDL <- (totC - (TG/5) - (HDL))
               
            }
            LDLvec[i] <- LDL
        }
        return(LDLvec)
    }


popCrAlbRat <- function (dataMat) {

    ## Function populates urinary creatinine / albumin ratio column from dataMat
    
    ## Creatinine Albumin ratio    
    creaVals <- dataMat[,"URXUCRSI"]
    albuVals <- dataMat[,"URXUMASI"]
    crAlbRat <- albuVals/(creaVals*1.1312*10^-4) ## NOTE: MAGIC NUMBERS (REF/UNITS)
    
    return(crAlbRat) 
} 


popLinAge <- function(linModel,dataMat,qDataMat) {

    ## This function takes a linear model (parameter file) and a dataMatrix
    ## containing all relevant data and generates a vector of linear ages.

    #######################
    ## Check data matrix ##
    #######################
    
    ## First, check that we have all columns we need
    dataFlags <- linModel[,"parType"]
    sexFlag <- linModel[,"sexFlag"]
    dataPars <- linModel[,"parName"][(dataFlags == "DATA") & (sexFlag == 1)]

    cat(paste("  > popLinAge: Data matrix passed validation - check for presence of all required columns\n"))
    ## Loop over the data matrix and make sure that we have each column
    nSEQs <- dim(dataMat)[1]
    errCode <- 0 
    for (nextCol in dataPars) {
        colLen <- length(dataMat[,nextCol])
        if (colLen != nSEQs) {
            cat(paste("  > ERROR! > Parameter:",nextCol,"does have:",colLen,"data - should be",nSEQs,"!\n"))
            errCode <- errorCode + 1
        }
    }
    if (errCode == 0) { cat(paste("  > popLinAge: Data matrix passed validation - looks like we have all data\n"))
    }

    ###################################################################
    ## Select relevant columns of data matrix - ensure correct order ##
    ###################################################################
    
    ## Now that we know we have all data columns, make selection mask
    allTerms <- colnames(dataMat) 
    dataMask <- match(dataPars,allTerms)  ## Records for each para the relevant col Nrs in dataMat 


    ###############################################
    ## Extract model parameters from model file  ##
    ###############################################
    cat(paste("  > popLinAge: Extract model parameters \n"))
    #########  Weights (betas)  and C1  ###########
    ## Get beta (weights) vectors for male and female SEQs
    ## Male
    beta_m <- linModel[,"betaVal"][linModel[,"sexFlag"]==1]
    nParas <- length(beta_m)-2 ## The last two values in this column are b0 (for chron age) and C1
    beta0_m <- beta_m[nParas+1] ## beta0_m is the weight of chron age in the model (for males) 
    C1_m <- beta_m[nParas+2] ## C1_m is the offset parameter in the model (for males) 
    beta_m <- beta_m[1:nParas] ## The rest are the parameter weights

    ## Female
    beta_f <- linModel[,"betaVal"][linModel[,"sexFlag"]==2]
    nParas <- length(beta_f)-2 ## The last two values in this column are b0 (for chron age) and C1
    beta0_f <- beta_f[nParas+1] ## beta0_f is the weight of chron age in the model (for females) 
    C1_f <- beta_f[nParas+2] ## C1_f is the offset parameter in the model (for females) 
    beta_f <- beta_f[1:nParas] ## The rest are the parameter weights

    cat(paste("  > popLinAge: Parameters extracted \n"))
    cat(paste("  > popLinAge: Extracting normalizing factors (healthy cluster) \n"))
    #########   medians and MAD of healthy cluster (normalize)  ##########
    ## For normalization we need median and MAD values for all parameters (from model)
    ## Note, these are sex specific
    ## Get median vectors
    ## Males
    medVec_m <- linModel[,"medVal"][(dataFlags == "DATA") & (sexFlag ==1)]
    madVec_m <- linModel[,"madVal"][(dataFlags == "DATA") & (sexFlag ==1)]
    ## Columns that will not be normalized have NAs here - so set med to 0 and MAD to 1 for those
    medVec_m[is.na(medVec_m)] <- 0
    madVec_m[is.na(madVec_m)] <- 1 
    
    ## Females
    medVec_f <- linModel[,"medVal"][(dataFlags == "DATA") & (sexFlag ==2)]
    madVec_f <- linModel[,"madVal"][(dataFlags == "DATA") & (sexFlag ==2)]
    ## Columns that will not be normalized have NAs here - so set med to 0 and MAD to 1 for those
    medVec_f[is.na(medVec_f)] <- 0
    madVec_f[is.na(madVec_f)] <- 1
    cat(paste("  > popLinAge: Parameters extracted \n"))
    
    #####################################################
    ## Main loop - Apply model to each SEQ in dataMat  ##
    #####################################################
   
    allSEQs <- dataMat[,"SEQN"]
    delAgeVec <- rep(0,nSEQs)
    i <- 0
    cat(paste("  > popLinAge: Starting loop over all SEQs in dataMat provided \n"))
    cat(paste("  > Processing: "))
    ## Now loop over the dataMatrix and extract the relevant data for each SEQ 
    for (nextSEQ in allSEQs) {
        i <- i + 1

        ## Get the line number of the dataMatrix
        nextLine <- match(nextSEQ,allSEQs)

        ## Check chron age and sex of next SEQ
        nextSex <- qDataMat[nextLine,"RIAGENDR"]
        nextAge <- qDataMat[nextLine,"RIDAGEYR"] ## Age in years 

        refAge <- round((dataMat[nextLine,"deltaPC_mAge_Test"]/12),3)
        
        ## Now extract vector of values for the linAge paras (in correct order) from dataMat for nextSEQ
        nextDatVec <- as.numeric(dataMat[nextLine,dataMask])
        
        ## Calculate delta age for male / female sub separately 
        ## Model equation: deltaBioAge = C1 + beta_0*chronAge+sum(beta_i*para_i)

        if (nextSex == 1) { ## Male SEQ

            ## Apply normalization into robust equivalent of z-score vs healthy reference
            ## Equation: (value_i - median_i)/MAD_i
            nextDatVec <- (nextDatVec - medVec_m)/madVec_m            
            
            ## Multiply the (normalized) data vector (nextDataVec) by the beta vector (beta_m/f)
            parXbeta <- nextDatVec*beta_m
            
            ## Delta age is exp(covsAll)
            delAge <- C1_m + beta0_m*nextAge*12 + sum(parXbeta) 
            
        }
        if (nextSex == 2) { ## Female SEQ

            ## Apply normalization into robust equivalent of z-score vs healthy reference
            ## Equation: (value_i - median_i)/MAD_i
            nextDatVec <- (nextDatVec - medVec_f)/madVec_f
            
            ## Multiply the (normalized) data vector (nextDataVec) by the beta vector (beta_m/f)
            parXbeta <- nextDatVec*beta_f
            
            ## Delta age is exp(covsAll)
            delAge <- C1_f + beta0_f*nextAge*12 + sum(parXbeta)
            
        }
        delAgeVec[i] <- delAge

        ## Update progress every 100 SEQs        
        if ((i %% 100)==0) {
            cat("*")
        }
    }
    cat(" DONE\n")
    cat(paste("  > popLinAge: bioAge delta determined for:",i,"out of",nSEQs,"subjects\n\n"))
    return(delAgeVec)
}

################################################################ 
## <<< END                  Function declarations             ##
################################################################



qDataFile <- "qDataMat_test.csv"   ## File containing a qData matrix
dataFile <- "dataMat_test.csv"     ## File containing a data matrix

## 1) Read in new qData (questionnaire and demographics) matrix
##############################################################
cat(" > Reading qData matrix (demographic and questionnaire data) ...\n")
qDataMat <- read.csv(qDataFile)
cat("   qData Matrix successfully read\n")
cat(paste("   Dimensions of qDataMatrix (SEQs qParameters): \n"))
cat(paste("     >",dim(qDataMat)),"\n")
cat("\n")

## 2) Read in new data matrix (clinical markers and parameters)
###############################################################
cat(" > Reading data matrix (clinical data and parameters) ...\n")
##dataMat <- read.csv("dataMat_linAge.csv")
dataMat <- read.csv(dataFile)
cat("   Data Matrix successfully read\n")
cat(paste("   Dimensions of dataMatrix (SEQs  data parameters):\n"))
cat(paste("     >",dim(dataMat)),"\n")
cat("\n")

## 3) Prep data matrix (calculate derived features, apply logs, digitize cot ...)
#################################################################################
## NOTE: NHANES IV is top coded age at 85, so anybody over the age of 85 will be called "85"
## Action: Drop any SEQ with age >= 85
## linAge was trained only on subjects 40 and above, so we should not use it for younger 
keepMe <- (qDataMat[,"RIDAGEYR"] >= 40 & qDataMat[,"RIDAGEYR"] < 85)

## only keep SEQs younger than 85
dataMat <- dataMat[keepMe,]
qDataMat <- qDataMat[keepMe,]

## 4) Prep data matrix (calculate derived features, apply logs, digitize cot ...)
#################################################################################

## Calculate derived features:
######################################
## Add fs scores to dataMatrix
cat("> Calculating derived quantities\n")
cat("  > Calculating fs scores")
fs1Score <- popPCFIfs1(qDataMat)
fs2Score <- popPCFIfs2(qDataMat) 
fs3Score <- popPCFIfs3(qDataMat)
dataMat<-cbind(dataMat,fs1Score,fs2Score,fs3Score)
cat("... done\n")

## Add creatinine to albumin ratio
cat("> Calculating creatinine to albumin ratio")
crAlbRat <- popCrAlbRat(dataMat)
dataMat<-cbind(dataMat,crAlbRat)
cat("... done\n")


## Add LDL value (from HDL/Chol/TRIGS)
cat("> Calculating LDL Cholesterol")
LDLV <- populateLDL(dataMat,qDataMat)
dataMat <- cbind(dataMat,LDLV)
cat("... done\n")

## Digitize cotinine
cat("  > Digitizing cotinine values into smoking intensities\n")
dataMat <- digiCot(dataMat)
cat("  > Done\n\n")

## Apply normalization:
######################
cat("> Applying parameter transformations\n")
## Take log of CRP
cat("  > Log transforming CRP\n")
dataMat[,"LBXCRP"] <- log(dataMat[,"LBXCRP"])

## Take log of BNP
cat("  > Log transforming SSBNP\n\n")
dataMat[,"SSBNP"] <- log(dataMat[,"SSBNP"])

## Read in model from file
cat("> Reading linAge model parameters from file: linAge_Paras.csv \n")
linAgePars <- read.csv("linAge_Paras.csv",sep=",")

## Calculate delta age from linear model - based on parameter file
cat("> Running linAge model for all subjects in dataMat, qDataMat \n")
delAge <- popLinAge(linAgePars,dataMat,qDataMat)

## Calculate linAge and add chronAge, age deltas and linAge to data matrix
cat("> Updating dataMat with chronAge, deltaAge and linAge for each subject \n")
chronAge <- qDataMat[,"RIDAGEYR"]
linAge <- chronAge + delAge
dataMat <- cbind(dataMat,chronAge,delAge,linAge)

cat("> Writing updated data matrix \n")
write.csv(dataMat,"dataMatrix_Normalized_With_Derived_Features_LinAge.csv")
cat("> DONE!\n\n")


