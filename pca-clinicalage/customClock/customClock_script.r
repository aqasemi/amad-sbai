## Example script illustrating training PCA-based linear clock using custom feature set
## >>> See: README.txt for more information...

## Load required libraries
library(survival)   ## Library for survival analysis
library(glmnet)     ## Optimizer for regression parameters 
library(pROC)       ## Receiver Operating Characteristic (ROC) figure
library(ggplot2)    ## Plotting library for nicer figures          

## Clear workspace
rm(list=ls())


######################################   FUNCTION DEFINITIONS   ############################################
## FUNCTIONS START >>> 

###############################   < READ, WRITE AND CLEAN DATA >  ##########################################

markIncsFromCodeBook<-function(codeBook) {

    ## Selects all columns marked as data ("1" in Data column) in codebook for inclusion in master dataset
    incFlags<-codeBook[,"Data"]   ## This column is "1" for include in dataset, 0 for not (comments or demo data) 
    incNames<-as.character(codeBook[,1]) 
    incList<-cbind(incNames,incFlags)
    return(incList)
}


dropCols<-function(dataSet,incList) {

    ## Simply drops everything not flagged with 1 in incList (Data)
    incInds<-which(incList[,2]==1)
    incTerms<-incList[,1][incInds]
    nTerms<-length(incTerms)
    
    ## The first column is always the subject number (SEQN) - add that back
    outData<-dataSet[,1]

    for (i in 2:nTerms) {
        
        ## loop over terms that have a "1" in column 2 of the incList, find those in
        ## dataSet and include in output dataSet
        nexTerm<-incTerms[i]
        nextCol<-which(colnames(dataSet)==nexTerm)
        colData<-dataSet[,nextCol]
        outData<-cbind(outData,colData)
        colnames(outData)[i]<-nexTerm
    }

    ## Name first column appropriately and return resulting dataset
    colnames(outData)[1]<-"SEQN"
    return(outData)
}


dropNAcolumns<-function(dataSet,pNAcut,incSwitch,verbose) {
        
    ## This takes a single cutoff fraction and drops all columns (features)
    ## that contain more NAs than allowed by the cutoff.
    ## However, if force include flag is set (==1) for a column, we will force the inclusion of feature

    nRows<-dim(dataSet)[1]
    ncols<-dim(dataSet)[2]
    forceFlags<-rep(0,nCols)

    ## If incSwitch IS set, we read them from the codebook
    if (incSwitch==1) {
        ## Read include flags from codebook
        codeBookFlags<-codeBook$ForceInc       
        ## Identify column terms that we cannot drop
        nForced<-sum(codeBookFlags) 
        forceIncTerms<-codeBook$Var[codeBookFlags==1]
        ## Now identify columns in dataset that need to be retained
        for (i in 1:nForced) {
            nextTerm<-forceIncTerms[i]
            forceThis<-which(colnames(dataSet)==nextTerm)
            ## Flip the respective forceFlag to 1 - this column cannot be dropped
            nrOfNAs<-sum(is.na(dataSet[,forceThis]))
            if (verbose) {
                print(paste("Applying force flag to:",nextTerm))
                cat("\t - this will include:\t",nrOfNAs,"\t NAs in:\t",nRows," ",round(nrOfNAs/nRows*100,2),"%\n")
            }
            forceFlags[forceThis]<-1            
        }
    }
    ## Now drop all columns with too many NAs
    naColSum<-colSums(is.na(dataSet))
    naColP<-naColSum/nRows

    ## Keep only those columns (features) for which naColP (number of NAs) is smaller than pNAcut
    keepCols<-naColP<pNAcut

    ## Finally, recover all columns that we decided to force (retain) 
    ## Merge keepCols (columns that will be kept due to cutoff) and forceFlags
    keepCols<-keepCols | (forceFlags == 1)
    dataSet<-dataSet[,keepCols]

    ## Print dimension of surviving matrix and list of surviving variables
    nrows<-dim(dataSet)[1]
    ncols<-dim(dataSet)[2]
    varNames<-colnames(dataSet)
    humNames<-varNames
    
    for (i in 2:length(varNames)) {
        varName<-varNames[i]
    }

    return(dataSet)
}
    
 
qDataMatGen<-function(masterData,incList) {

    ## Loop over masterData and keep any column that has a zero in the "data" column
    allTerms<-colnames(masterData)
    nTerms<-dim(masterData)[2]   ## number of total terms (columns) in masterData
    nIncFlags<-dim(incList)[1]   ## number of terms in codebook - for which we know include flags
    
    ## The first column of the qDataMatrix has to be SEQN number - add these first
    qDataNames<-"SEQN"
    qDataMatrix <- masterData[,1]
    
    ## Loop over all terms (columns) in masterData - extract one term (column) at a time
    for (i in 2:nTerms) {    
        ## look at the next term in the data and get the respective flag from the incList
        nextTerm<-allTerms[i]

        ## Now loop over all terms in the incList and get the flag for the current term
        flag<-0   ## Graceful default is 0 - not include
        for (j in 1:nIncFlags) {
            if (incList[j,1]==nextTerm) { 
                ## Read the inc flag (second entry of that column) and return it
                flag <- incList[j,2]
            }
        }
        if (flag == 0) {
            ## If include == 0, we will include that parameter in the qDataMatrix 
            qDataColumn <- masterData[,i]    ## Keep the current column for inclusion to qDataMatrix 
            qDataMatrix<-cbind(qDataMatrix,qDataColumn)  ## Add current column to qDataMatrix 
            qDataNames<-c(qDataNames,nextTerm)  ## Also keep the current column name (nextTerm) as column name 
        }
    }
    colnames(qDataMatrix) <- qDataNames  ## Update all column names  
    return(qDataMatrix)  ## Return the matrix 
}


getNonNARows<-function(dataSet) {

    ## Identify rows that contain NAs and drop them by only retaining those that do not
    ## sums over NAs are NA so only rows with no (zero) NAs return !is.na
    keepRows<-(rowSums(is.na(dataSet))==0)
    return(keepRows)
}



#########################  < CALCULATING DERIVED FEATURES FROM DATA >  #####################################

popPCFIfs1 <- function(qDataMat) {
        ## This will calculate our frailty index / disease and comorbidity index for each subject
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

        ## Binary yes/no decision vector 
        binVec <- cbind((BPQ020==1),((DIQ010==1) | (DIQ010==3)),(KIQ020==1),(MCQ010==1),(MCQ053==1),(MCQ160A==1),(MCQ160C==1),(MCQ160D==1),(MCQ160E==1),(MCQ160F==1),(MCQ160G==1),(MCQ160I==1),(MCQ160J==1),(MCQ160K==1),(MCQ160L==1),(MCQ220==1),(OSQ010A==1),(OSQ010B==1),(OSQ010C==1),(OSQ060==1),(PFQ056==1),(HUQ070==1))

    sumOverBinVec <- rowSums(binVec)/22 
    return(sumOverBinVec)
}


popPCFIfs2 <- function(qDataMat) {
        
    HUQ010 <- qDataMat[,"HUQ010"]
    HUQ020 <- qDataMat[,"HUQ020"]
    HUQ010[is.na(HUQ010)] <- 3
    HUQ020[is.na(HUQ020)] <- 3

    ## If sick/feeling bad, get score of 2 to 4 - if getting worse -> get 2x modifier
    ## if getting better -> 1/2 modifier 
    aVec <- ((HUQ010==4)*2+(HUQ010==5)*4)
    dVec <- (1-(HUQ020==1)*0.5+(HUQ020==2))
    fScore <- aVec*dVec
        
    return(fScore)
}


popPCFIfs3 <- function(qDataMat) {

    ## This basically codes NHANES HUQ050: "Number times received healthcare over past year"
    HUQ050  <- qDataMat[,"HUQ050"]
    HUQ050[is.na(HUQ050)] <- 0
    HUQ050[(HUQ050==77)] <- 0 ## Comment codes ("Refused")
    HUQ050[(HUQ050==99)] <- 0 ## Comment codes ("Do not know")
    return(HUQ050)
}


populateLDL <- function (dataMat,qDataMat) {                                                                           
                                                                                                                            
        ## This function will calculate LDL and adds it to the dataMatrix
        ## LDL - calculated from:
        ##    Variable: LBDTCSI	        Total Cholesterol (mmol/L)
        ##    Variable: LBDHDLSI	HDL (mmol/L)
        ##    Variable: LBDSTRSI	Triglycerides (mmol/L)
        ## Formula:  LDL-C=(TC)–(triglycerides/5)– (HDL-C). 
        ## NOTES: Can be inaccurate if triglycerides are very high (above 150 mg/dL)
        
        nSubs <- dim(dataMat)[1]
        
        ## Extract all relevant variables from data matrix 
        totCv <- dataMat[,"LBDTCSI"]
        HDLv <- dataMat[,"LBDHDLSI"]
        triGv <- dataMat[,"LBDSTRSI"]
        seqVec <- dataMat[,"SEQN"]
        LDLvec <- rep(0,nSubs)
        
        ## Loop over all subjects and update LDL
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

                            
#############################  < DATA SELECTION - ROWS / SUBJECTS >  ####################################

selectAgeBracket<-function(qMat,ageCutLower,ageCutUpper) {

    ## Apply a age bracket to dataset - only retain samples between upper and lower age limit
    keepRows<-((qMat[,"RIDAGEYR"]>=ageCutLower) & (qMat[,"RIDAGEYR"]<=ageCutUpper))
    return(keepRows)
}


nonAccidDeathFlags <-function(qMat) {

    ## Here we will return keep flags for all subjects who die of non-accidental deaths
    ## The cause of death (leading) is recorded (if known) in the questionnaire data matrix
    ## qDatMat in the "UCOD_LEADING" column
    ## Possible values in "UCOD_LEADING" are:
    ## 001 = Disease of the heart
    ## 002 = Malignant neoplasm
    ## 003 = Chronic lower respiratory disease
    ## 004 = Accidents and unintentional injuries
    ## 005 = Cerebrovascular disease
    ## 007 = Diabetes
    ## 008 = Influenza and pneumonia
    ## 009 = Nephritis, kidney issues
    ## 010 = All other (residuals)
    ## NA  = no info (the vast majority of cases)

    ## Extract cause of deaths
    causeOfDeath <- qMat[,"UCOD_LEADING"]
    ## Then drop NAs (turn into zeros) 
    causeOfDeath[is.na(causeOfDeath)]<-0
    keepFlags <- causeOfDeath!=4
    
    return(keepFlags) 
                
}


#############################    MATH AND ANLYSIS FUNCTIONS    #######################################

normAsZscores<-function(dataSet) {

    ## Normalize data by column average for each column independently
    nRows<-dim(dataSet)[1]
    nCols<-dim(dataSet)[2]

    ##  Calculate means and SDs for each column (feature) 
    colAvs<-colMeans(dataSet)
    colSDs<-apply(dataSet,2,sd)
    
    ## Make normalized matrix by turning each value into z-score
    dataMatN<-matrix(0,nRows,nCols)

    ## There is certainly a more elegant way of doing this - but for hackability, lets just do a
    ## simple loop for now - column 1 is still the subject seq number - will not be normalized
    dataMatN[,1]<-dataSet[,1]
    colnames(dataMatN)<-colnames(dataSet)

    ## Loop over all columns - starting from column 2 (first data col) 
    for (col in 2:nCols) {
        ## Get mean feature value for current column 
        mean<-colAvs[col]
        sd<-colSDs[col]
        ## Loop over all rows in current column and normalize each value by column mean
        for (row in 1:nRows) {
        zScore<-(dataSet[row,col]-mean)/sd
        
        ## Store normalized and log2 fold changes (vs. column average) in new matrices
        dataMatN[row,col]<-zScore
        }
    }
    return(dataMatN)
}


getSurvTime <- function (qMatrix) {

    ## Function to calculate survival time between enrollment and end of follow up
    ## Get the age (in month) at time of initial screen

    ## For those individuals who died before the cutoff date in 2019, we have
    ## information on time between survey and death - for survivors, the entry is time
    ## between the initial exam and the end of the follow up

    ## NOTE: THIS IS REALLY NOT SURVIVAL TIME BUT TIME TO FOLLOW UP - interpret with eventFlags! 
    survTimes<-qMatrix[,"PERMTH_EXM"]
    return (survTimes)

    }


getEventVec <- function (qMatrix, cause) {

    ## Read qDataMatrix and determine if individual died during study period or was censored
    ## that is, survived beyond the end of the study ... 

    if (cause == 0) {  ## IF cause is 0, we do not care what people died from and report all deaths 
        eventFlags<-qMatrix[,"MORTSTAT"] 
        return (eventFlags)
    }
    if (cause !=0) {  ## If cause is > 0, we will report only specific causes of death
        eventFlags<-qMatrix[,"MORTSTAT"]
        CODFlags <- qMatrix[,"UCOD_LEADING"]
        
        if (cause == 1) { ## Heart disease deaths only
            countThese <- (CODFlags==1)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 2) { ## Cancer deaths only
            countThese <- (CODFlags==2)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 3) { ## COPD deaths only 
            countThese <- (CODFlags==3)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 4) { ## Accident deaths only 
            countThese <- (CODFlags==4)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        
        if (cause == 5) { ## Stroke deaths only 
            countThese <- (CODFlags==5)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 6) { ## Deaths directly from AD only 
            countThese <- (CODFlags==6)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 7) { ## Deaths directly form diabetes only 
            countThese <- (CODFlags==7)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 8) { ## Deaths from influenza and pneumonia  
            countThese <- (CODFlags==8)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 9) { ## Deaths from kidney issues  
            countThese <- (CODFlags==9)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        ## We can also specify some causes that are combinations or exclusions of others
        if (cause == 10) { ## All NON CVD (not MCI, not stroke) deaths only  
            countThese <- ((CODFlags!=1) & (CODFlags!=5)) 
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        
        if (cause == 11) { ## All non accidental deaths only   
            countThese <- ((CODFlags!=4)) 
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }

        if (cause == 12) { ## All CVD-related deaths - including stroke   
            countThese <- ((CODFlags==1 ) | (CODFlags==5)) 
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
    }
}


makeSurvObject <- function (qMatrix,causeOfDeath) {

    ## Take survival times and censor vector and make a survival object 

    ## First, get event flags from qMatrix - subjects that died have 1, those that survived have 0 in here
    eventFlags <- getEventVec(qMatrix,causeOfDeath)

    ## Then, get survival times (either time between exam and death (if dead) or time to end of follow up (alive)
    times <- getSurvTime(qMatrix)

    ## Now make a survival object (survival library)
    survObj<-Surv(times,eventFlags) 
    
    return(survObj) 
    
}

    
cloneCohort <- function (dataMat0, noiseLevel) {

    ## This will simply take a data matrix and add stochastic noise at level (SD) of noiseLevel
    ## to every column (feature) before returning the "cloned" matrix 

    nFeats <- dim(dataMat0)[2]
    mSEQs <- dim(dataMat0)[1]
    for (nextFeat in 2:nFeats) {

        ## Make vector of normal distributed noise  
        erFac <- rnorm(mSEQs)*noiseLevel
        erVec <- erFac* dataMat0[,nextFeat]

        ## Add to column and update
        dataMat0[,nextFeat] <- dataMat0[,nextFeat] + erVec
    }
    return(dataMat0)
}
        
        
calcBioAge <- function (coxModelNew,nullModel,dataTable) {
   
    ## This will take the coxModel plus the input data table (covariates used for the cox model)
    ## it will then loop over the data table and calculate the delta ages for each individual    
    ## The cox model assumes that the hazard function hi(t) for each subject i
    ## can be broken down into log additive terms according to the linear (lm) model
    ## plus the universal time-dependent (follow up time) term  h0(t)             
    ## e.g. here: hi(t) = h0(t)*exp(beta1*startAge + beta2*x2 + beta3*x3 + beta4*x4)
    
    ## First extract maximum likelihood betas from full coxModel
    betasCOX <- coxModelNew$coefficients
    betasNull <- nullModel$coefficients
    
    ## We know that h is the mortality function according to gompertz - we can infer that:             
    ##     beta1*ageStart == ln(2)/MRDT*ageStart
    ## <=> beta1 == ln(2)/MRDT
    ## <=> MRDT == ln(2)/beta1
    ##
    betaOne <- betasNull[1]
    MRDTfit <- round(log(2)/betaOne,2)
    riskMod <- predict(coxModelNew,newdata=dataTable,type="risk")
    riskNull <- predict(nullModel,newdata=dataTable,type="risk")
    
    logRiskRatio <- log(riskMod/riskNull) 
    ageBioDelta <- logRiskRatio/log(2)*MRDTfit   
    return(ageBioDelta)           
}


varNH2Hum <- function(varName) {

    ## Simply look up variable name in codebook and return human readable entry 
    varLine <- which(codeBook[,"Var"]==varName)
    
    if (identical(varLine,integer(0))) {     ## No such line
        words <- paste("Variable: ",varName," is not found in the codebook\n")
        humanDef <- "nope"
        
    } else {
                       
        ## Found a line - dump data 
        humanDef <- codeBook[varLine,"Human"]
        data <- codeBook[varLine,"Data"]
        force <- codeBook[varLine,"ForceInc"]    
        words <- paste("Variable: ",varName," has Data flag:",data,"and Force Inclusion flag:",force,".\nIt has human meaning: <",humanDef,">\n")

    }
    cat(words)
    return(humanDef)
}
        
## <<<  END FUNCTION DEFINITIONS





#####################################################################################################################
####                                     >>> MAIN CUSTOM CLOCK SCRIPT <<<                                        #### 
#####################################################################################################################
## Get source file name and report start


###################################
## I) DATA FILES AND DATA IMPORT ##
###################################
cat("\n\n\n\n#################################################################################\n") 
cat("## >>>  Please see: README.txt for instructions on how to use this script  <<< ## \n")
cat("#################################################################################\n") 
sourceFile <- parent.frame(2)$ofile
cat("SOURCE: [",sourceFile,"] \nSTART >>>\n\n")

## NOTE: Features with "1" in the "data" column in codebook can be used as features
##       Features with "1" in the "forceInc" column in codebook WILL be used as features
##       See README.txt for details ...
codeBookFile <- "codebook_custom_linAge.csv"
cat(paste("> Please modify codebook file: [",codeBookFile,"] to select features !\n>"))
readline(" > DONE ?    (((   Press key to continue   )))    > ")


cat("\nI) Reading data and configuration files\n")
cat("#######################################\n") 


## First, read the parameter file containing some parameters that can be changed
paraFile <- "paraInit.csv"
cat("> Reading codebook:",paraFile,"... ")        
paras <- read.csv(paraFile,sep=",",header=TRUE) 
cat("Done\n\n")


## Now use this to fix some important parameters
## See paraInit.csv for summary of what these mean 
errLevl <- paras[(paras[,"pName"]=="errorLevel"),"pValue"]   
pNAcut <- paras[(paras[,"pName"]=="NAcutOffpercent"),"pValue"]
svCutP <- paras[(paras[,"pName"]=="PCcutOffpercent"),"pValue"]
ageLower <- paras[(paras[,"pName"]=="lowerAgeLimit"),"pValue"]  
ageUpper <- paras[(paras[,"pName"]=="upperAgeLimit"),"pValue"]
useDerived <- paras[(paras[,"pName"]=="derivedFeatFlag"),"pValue"] 
verbose <- 0     ## Sets level of verbosity for some functions - 0 means not very

## Second, read in csv of total NHANES (continuous) data - here we load the year 1999 with most
## lab data included
dataFileName <- "mergedData_PA.csv"
cat(paste("> Reading data: [",dataFileName),"] ... ")    
masterData <- read.csv(dataFileName)
cat("Done\n")

## Codebook file
cat("> Reading codebook:",codeBookFile,"... ")        
codeBook <- read.csv(codeBookFile)
cat("Done\n\n")
## The codebook file contains the Variable names in NHANES 99/00 format (Var) and human readable (Human)
## The codebook file  also contains a flag (Demo/Exam...) coding for the type of data - the flags are:
##     DEMO: Demographic data
##     Q        : Questionnaire
##     E        : Medical Examination
##     LAB      : Clinical Laboratory
##     MORTALITY: Mortality / Survival and cause of death linkage
## Finally, the codebook contains a flag indicating  numerical data (Data) and forced inclusion (ForceInc)
## These flags can be 1 (yes) or 0 (no)
## Set both Data and ForceInc to 1 for variables to be used for custom clock...


##############################
## II)  PREPARE DATA MATRIX ##
##############################
    
###############################
## II.i) FILTER INPUT MATRIX ##
###############################
cat("\nII) Selecting and cleaning data\n")
cat("###############################\n")    

cat("> Splitting data matrix ... ")
## Drop non-data columns from master data based on include flags in the codebook
cat(" selecting data ...  ")
incList <- markIncsFromCodeBook(codeBook)
dataMat <- dropCols(masterData,incList)  ## Main data matrix for clock

## Now we make a questionnaire Data matrix - everything OTHER than the numerical / clinical data
## qDataMat will include anything that is NOT flagged as "data" in the codebook     
cat(" selecting qData ... ")
qDataMat <- qDataMatGen(masterData,incList)   ## NOTE: This is pretty much the same as dropCols for dataMatrix ...
cat("Done\n")


######################################
## II.ii) POPULATE DERIVED FEATURES ##     
######################################
## Only gets done if useDerived == 1, skipped else
if (useDerived) {
    cat("> Populating derived features ... ")
    cat(" fs scores ...")
    fs1Score <- popPCFIfs1(qDataMat)
    fs2Score <- popPCFIfs2(qDataMat) 
    fs3Score <- popPCFIfs3(qDataMat)
    dataMat <- cbind(dataMat,fs1Score,fs2Score,fs3Score)


    ## Check if we have what we need for LDL
    if(((sum(colnames(dataMat)=="LBDTCSI")==1)*
        (sum(colnames(dataMat)=="LBDHDLSI")==1)*
        (sum(colnames(dataMat)=="LBDSTRSI")==1))==1) {
        
        ## LDL values     
        cat(" LDLV ...")
        LDLV <- populateLDL(dataMat,qDataMat)
        dataMat <- cbind(dataMat,LDLV)
    }
    
    if(((sum(colnames(dataMat)=="URXUCRSI")==1)*(sum(colnames(dataMat)=="URXUMASI")==1))==1) {
        ## Albumin Creatinine ratio
        cat(" Albumin Creatinine ratio ... ")
        creaVals <- dataMat[,"URXUCRSI"]
        albuVals <- dataMat[,"URXUMASI"]
        crAlbRat <- albuVals/(creaVals*1.1312*10^-4)
        dataMat<-cbind(dataMat,crAlbRat)
        cat("Done\n")
    }
}

## Return dimensions of original input data matrix before cleanup functions
nRows<-dim(dataMat)[1]
nCols<-dim(dataMat)[2]
cat(paste("\n>>> Raw data matrix has ",nRows," rows (subjects) and ",nCols," columns (features) <<<\n\n"))

## We also need to drop all subjects for which we have no information on age
cat("> Removing subjects with missing age data ... ")
subSansAge <- which(is.na(qDataMat[,"RIDAGEEX"]))
dataMat <- dataMat[-subSansAge,]
qDataMat <-qDataMat[-subSansAge,]
cat("Done \n")

    
####################################################################
## II.iii) REFINE COHORT BY FURTHER DEMOGRAPHIC AND LAB CRITERIA  ##           
####################################################################
## Drop all accidental death cases
cat("> Removing accidental deaths ... ")
keepRows <- nonAccidDeathFlags(qDataMat) 
dataMat <- dataMat[keepRows,]
qDataMat <- qDataMat[keepRows,]
cat("Done\n")

## Remove individuals below the age of ageCut - also remove individuals over 84 
## as the age data is top-coded at 84 (e.g. 100 is recorded as 85)
cat("> Applying age filter: [",ageLower,",",ageUpper,"] years  ... ")
keepRows <- selectAgeBracket(qDataMat,ageLower,ageUpper)
dataMat <- dataMat[keepRows,]
## NOTE: Any time that we drop rows (subject), we have to also drop the same rows
## from demographic data and update the sequence data:
qDataMat <- qDataMat[keepRows,]
cat("Done\n")

## Next, we need to remove columns (features) with excessive number of missing values
cat("> NA percentage threshold for dropping feature is set to:",pNAcut*100,"%\n")
cat("> Dropping features with more NAs than threshold ... ")
dataMat<-dropNAcolumns(dataMat,pNAcut,1,verbose)
cat("Done\n")
cat("> Dropping subjects with NAs  ... ")
## Drop all subjects with missing values from the dataset
keepRows <- getNonNARows(dataMat) 
dataMat <- dataMat[keepRows,]
## Also need to again update the demographic matrix to remove the same people
qDataMat <- qDataMat[keepRows,]
cat("Done\n\n")


## See what is left of our input data matrix after cleanup functions
nRows <- dim(dataMat)[1]
nCols <- dim(dataMat)[2]
cat(paste(">>> Final data matrix has ",nRows," rows (subjects) and ",nCols," columns (features) <<< \n"))

## Here we clone the whole cohort with added noise
dataMat_stoch <- cloneCohort(dataMat,errLevl)

#################################
## III) NORMALIZATION OPTIONS  ##
#################################    
    
############## PICK NORMALIZATION OPTION HERE BY CALLING FUNCTION ################
cat("\nIII) Normalizing training and testing data\n")
cat("##########################################\n")    
cat("> Normalizing as z-score  ... ")
dataMatNorm <- normAsZscores(dataMat)
dataMatNorm_stoch <- normAsZscores(dataMat_stoch)     
cat("Done\n")


cat("> Splitting data into training (99/00 wave) and testing (01/02 wave) subsets ... ")
inputMat <- dataMatNorm[,2:nCols] ## Drop SEQn (subject number) from input data for PCA/SVD
inputMat99 <- inputMat[(qDataMat[,"yearsNHANES"]==9900),]  ## Only use 99/00 rows - all columns
inputMat01 <- inputMat[(qDataMat[,"yearsNHANES"]==102),]  ## Only use 01/02 rows - all columns
cat("Done\n")

## Also extract testing matrix from dataMat_stoch
inputMat01_stoch <- dataMatNorm_stoch[(qDataMat[,"yearsNHANES"]==102),2:nCols]  

###############################################################
## IV)  DIMENSIONALITY REDUCTION / COORDINATE TRANSFORMATION ##
###############################################################

#################################
## IV.i)  Do the basic PCA/SVD ##
#################################
cat("\nIV) Dimensionality reduction by SVD\n")
cat("###################################\n")    
cat("> Calculating SVD and transforming training set into PC coordinates ... ")

## Now do the SVD ONLY for the 99/00 cohort    
svdData99 <- svd(inputMat99)    
    
## Extract diagonal matrix of singular values as well as right (V) and left (U) singular vectors
diagDat99 <- diag(svdData99$d)
uMatDat99 <- svdData99$u
vMatDat99 <- svdData99$v

## uMat (left singular vector) is of dimension nrOfSamples x nrOfSVDs
mSamples99 <- dim(uMatDat99)[1]
nSVs99 <- dim(uMatDat99)[2]

## Make data matrix of training set in PC coordinates derived from training data only 
pcMat99 <- uMatDat99 %*% diagDat99
cat("Done\n")

cat("> Projecting testing data into PC coordinates of training data ... ")
## Project testing set into the same PC coordinates for dataMat and dataMat_stoch
mSamples01 <- dim(inputMat01)[1]
pcMat01 <- matrix(0,mSamples01,nSVs99)  ## Testing data matrix
pcMat01_stoch <- matrix(0,mSamples01,nSVs99)  ## Testing data matrix with noise   
## Doing loop to calculate coordinates for samples in terms of PCs
for (sample in 1:mSamples01) {
    ## Current sample is current row of data (input) matrix
    curSample <- inputMat01[sample,]
    curSample_stoch <- inputMat01_stoch[sample,]
    ## Now loop over all nSVs and determine
    for (pcNr in 1:nSVs99) {
        ## current PC vector is the column
        curPC <- vMatDat99[,pcNr]
        coord <- curSample %*% curPC
        pcMat01[sample,pcNr]<-coord

        ## Same for matrix with noise added
        coord_stoch <-curSample_stoch %*% curPC
        pcMat01_stoch[sample,pcNr]<-coord_stoch
    }
}
cat("Done\n")

## pcMat99 are the SVD coordinates for the 9900 cohort - in SVD coordinates from 9900 cohort only
## pcMat01 are the SVD coordinates for the 0102 cohort - in SVD coordinates from 9900 cohort only
## Merge dataset - express BOTH 9900 and 0102 cohorts in SVD coordinates from 9900 cohort   
cat("> Merging PC data for both training and testing data ... ")
pcDatMat <- rbind(pcMat99,pcMat01)    
colnames(pcDatMat) <- paste("PC",1:nSVs99,sep="")
colnames(pcMat01_stoch) <- paste("PC",1:nSVs99,sep="")
cat("Done\n")

#######################################
## IV.ii)   DIMENSIONALITY REDUCTION ##
#######################################
## scree[n] * 100 is the percent explained by the nth singular vector - use this to truncate data
## at the point where the nth SV explains less than svCutP % of total variance - that is, where:
## scree[n] becomes less than svCutP/100
cat("> Calculating scree plot  ... ")
scree <- svdData99$d^2/sum(svdData99$d^2)
cat(" determining PCA cutoff ... ")
svCut <- which(scree<svCutP/100)
svCut <- min(svCut,nSVs99)  ## If no cutoff, use all SVs
cat("Done\n")

## Draw scree plot (pdf)
screeFile <- "scree.pdf"
cat("> Writing out scree plot: [",screeFile,"] ... ")
pdf(screeFile)
words <- paste("Scree Plot\n Cutoff:",svCutP,"% (blue line) at PC",svCut, "(red line)")
plot(scree*100,xlab = "PC Nr." ,ylab = "Variance explained (%)", type="l", col="gray", main=words, lwd=2)
points(scree*100, pch=16, col="black")
abline(v=svCut, lwd=2, lty="dashed", col="red")
abline(h=scree[svCut]*100, lty="dashed", lwd=2, col="blue")
dev.off()
cat("Done\n")

cat("> Reducing dimensionality by dropping dimensions (PCs) explaining less than",svCutP,"% of variance. \n")
## Truncate the dataMatrix at this point - dropping all higher SVs / PCs
pcDatMat<-pcDatMat[,1:svCut[1]]
maxPC <- svCut[1]
cat("> Dropped PCs beyond PC Nr.",maxPC," ... ")
cat("Done\n")


############################
## V)  CLOCK CONSTRUCTION ##
############################
cat("\nV) Building clock based on 99/00 wave\n")
cat("#####################################\n")    

##############################################
## V.i)  EXTRACT DEMO AND SEX OF INPUT DATA ##
##############################################
## We are using 99/00 NHANES wave as training set and 01/02 as testing est
trainSam <- (qDataMat[,"yearsNHANES"]==9900)
testSam <- (qDataMat[,"yearsNHANES"]==102)

## Extract demographics (qDataMat) for training and testing set 
demoTest <- qDataMat[testSam,]
demoTrain <- qDataMat[trainSam,]

## Then get age at time of examination  - this is always the first covariate
initAgeTrain <- demoTrain[,"RIDAGEEX"] 
initAgeTest <- demoTest[,"RIDAGEEX"] 

## Extract sex flag - 1 male, 2 female for testing and training 
sexTest <- qDataMat[testSam,"RIAGENDR"]
sexTrain <- qDataMat[trainSam,"RIAGENDR"]

## Extract ID of all training-set subjects
selTrain <-demoTrain[,1]    ## Here we could drop columns with NAs for instance     
selTest <- demoTest[,1]    

## Split the PCA matrix into test and train matrices 
xTrainPCA <- pcDatMat[trainSam,]  
xTestPCA <- pcDatMat[testSam,]

############################################################
## V.ii) MAKE COVARIATES FOR TRAIN/TEST COX PH MODELs     ##
############################################################
## Training
coxCovsTrain <- cbind(initAgeTrain,xTrainPCA,sexTrain)
colnames(coxCovsTrain)[1] <- "chronAge"  ## The variable names need to be fixed for the cox model function 
colnames(coxCovsTrain)[(maxPC+2)] <- "sex"
coxCovsTrain <- as.data.frame(coxCovsTrain)

## Testing
coxCovsTest <- cbind(initAgeTest,xTestPCA,sexTest) 
colnames(coxCovsTest)[1] <- "chronAge"
colnames(coxCovsTest)[(maxPC+2)] <- "sex"
coxCovsTest <- as.data.frame(coxCovsTest)

## Testing data with noise
coxCovsTest_stoch <- cbind(initAgeTest,pcMat01_stoch,sexTest) 
colnames(coxCovsTest_stoch)[1] <- "chronAge"
colnames(coxCovsTest_stoch)[(maxPC+2)] <- "sex"
coxCovsTest_stoch <- as.data.frame(coxCovsTest_stoch)


## SPLIT INTO MALE AND FEMALE SETS    ##
## Females
##########
testUseF <- demoTest[,"RIAGENDR"]==2
trainUseF <- demoTrain[,"RIAGENDR"]==2
## Female COX PH covariates
coxCovsTrainF <- coxCovsTrain[trainUseF,]
coxCovsTestF <- coxCovsTest[testUseF,]
## Female survival objects
survObjTrainF <- makeSurvObject(demoTrain,0)[(demoTrain[,"RIAGENDR"]==2)] 
## If any of these returns NA because of missing data, we need to drop that subject
keep <- which(!is.na(survObjTrainF))
survObjTrainF <- survObjTrainF[keep]
coxCovsTrainF <- coxCovsTrainF[keep,]
## Female survival object for testing set
survObjTestF <- makeSurvObject(demoTest,0)[(demoTest[,"RIAGENDR"]==2)]


## Males
########
testUseM <- demoTest[,"RIAGENDR"]==1
trainUseM <- demoTrain[,"RIAGENDR"]==1
## Male COX PH covariates
coxCovsTrainM<-coxCovsTrain[trainUseM,]
coxCovsTestM<-coxCovsTest[testUseM,]
## Male survival objects
survObjTrainM <- makeSurvObject(demoTrain,0)[(demoTrain[,"RIAGENDR"]==1)] 
## If any of these returns NA because of missing data, we need to drop that subject
keep <- which(!is.na(survObjTrainM))
survObjTrainM <- survObjTrainM[keep]
coxCovsTrainM <- coxCovsTrainM[keep,]
## Male survival object for testing set
survObjTestM <- makeSurvObject(demoTest,0)[(demoTest[,"RIAGENDR"]==1)]

## Testing data with noise
##########################
coxCovsTestF_stoch <- coxCovsTest_stoch[testUseF,]
coxCovsTestM_stoch <- coxCovsTest_stoch[testUseM,]


##################################################
## V.iii) OPTIMIZE PCs TO INCLUDE USING GLMNET  ##
##################################################
## Use lasso regression to identify parameters to include in coxph model
cat("> Optimizing model weights by glmnet ... ")

## Females 
##########
cat("Females ... ")
xF <- as.matrix(coxCovsTrainF)
cvfitF <- cv.glmnet(xF, survObjTrainF , alpha=1, family = "cox", type.measure = "C")
laMinF <- cvfitF$lambda.min
lamSEF <- cvfitF$lambda.1se
lamUseF <- mean(laMinF,lamSEF)

fitF <- glmnet(coxCovsTrainF, survObjTrainF , alpha=1, family = "cox")
## Use only parameters that have non-zero weight at optimal lambda ...
useParF <- (as.numeric(coef(fitF,s=lamUseF))!=0)

## Males
########
cat("Males ... ")
xM <- as.matrix(coxCovsTrainM)
cvfitM <- cv.glmnet(xM, survObjTrainM , alpha=1, family = "cox", type.measure = "C")
laMinM <- cvfitM$lambda.min
lamSEM <- cvfitM$lambda.1se
lamUseM <- mean(laMinM,lamSEM)
fitM <- glmnet(coxCovsTrainM, survObjTrainM , alpha=1, family = "cox")

## Use only parameters that have non-zero weight at optimal lambda ...
useParM <- (as.numeric(coef(fitM,s=lamUseM))!=0)
cat("Done\n")


############################
## V.iv) FIT FINAL MODEL  ##
############################
## Write out formulas for glmn
cat("> Constructing model formula ... ")
cat("Females ... ")
formF <- as.formula(paste("survObjTrainF ~", paste(factor(colnames(coxCovsTrainF)[useParF]),collapse=" + ")))
cat("Males ... ")
formM <- as.formula(paste("survObjTrainM ~", paste(factor(colnames(coxCovsTrainM)[useParM]),collapse=" + ")))
cat("Done\n")

## Dump formulas to screen for inspection 
cat("\nModel equation (females):\n>>>")
print(formF)
cat("<<< \n\n")
cat("Model equation (males):\n>>>")
print(formM)
cat("<<< \n\n")

## Now make coxph models for prediction
cat("> Fitting final models ... ")
## Females 
cat("Females ... ")
nullModelF <- coxph(survObjTrainF ~ chronAge, data=coxCovsTrainF)   
coxModelF <- coxph(formF, data=coxCovsTrainF)   
## Males
cat("Males ... ")
nullModelM <- coxph(survObjTrainM ~ chronAge, data=coxCovsTrainM)   
coxModelM <- coxph(formM, data=coxCovsTrainM)   
cat("Done\n")


###############################################################
## VI) CALCULATE BIOAGES FOR TESTING SET AND EVALUATE CLOCK  ##
###############################################################
cat("> Calculating BioAges for test data based on custom clocks ... ")
## BioAge deltas and BioAge
cat("Females ... ")
delBioAgeTestF <- calcBioAge(coxModelF,nullModelF,coxCovsTestF)
bioAgeTestF <- coxCovsTestF[,"chronAge"] + delBioAgeTestF
cat("Males ... ")
delBioAgeTestM <- calcBioAge(coxModelM,nullModelM,coxCovsTestM)
bioAgeTestM <- coxCovsTestM[,"chronAge"] + delBioAgeTestM
cat("Done\n")

## Also calculate bioAge for data with added stochastic noise
delBioAgeTestF_stoch <- calcBioAge(coxModelF,nullModelF,coxCovsTestF_stoch)
delBioAgeTestM_stoch <- calcBioAge(coxModelM,nullModelM,coxCovsTestM_stoch)
bioAgeTestF_stoch <- coxCovsTestF_stoch[,"chronAge"] + delBioAgeTestF_stoch
bioAgeTestM_stoch <- coxCovsTestM_stoch[,"chronAge"] + delBioAgeTestM_stoch

## Sort BA estimates back into the (mixed sex) testing data matrix 
nTest <- dim(demoTest)[1]
bioAge <- rep(0,nTest)
chrAge <- demoTest[,"RIDAGEEX"]
bioAge_stoch <- rep(0,nTest)
phenoAge <- demoTest[,"phenoAge"]

## Sort by SEQN
SEQnF <- demoTest[testUseF,"SEQN"]
SEQnM <- demoTest[testUseM,"SEQN"]
bioAge[!is.na(match(demoTest[,"SEQN"],SEQnF))] <- bioAgeTestF
bioAge[!is.na(match(demoTest[,"SEQN"],SEQnM))] <- bioAgeTestM
bioAge_stoch[!is.na(match(demoTest[,"SEQN"],SEQnF))] <- bioAgeTestF_stoch
bioAge_stoch[!is.na(match(demoTest[,"SEQN"],SEQnM))] <- bioAgeTestM_stoch

## Read PhenoAge_stoch (phenoAge calculated after adding noise to dataMatrix)
PA <- qDataMat[,"phenoAge"]
PA_stoch <- qDataMat[,"phenoAge_stoch"]
## Remove NA values from PA data
keep <- which(!is.na(PA))
PA <- PA[keep]
PA_stoch <- PA_stoch[keep]

## Finally, here we compare new custom clock BAs to PhenoAge BAs
cat("\nVI) Compare custom clock to BA estimate from PhenoAge \n")
cat("#####################################################\n\n")    
cat("> 1) ROC figure, comparing custom clock to PhenoAge\n")
cat("---------------------------------------------------\n")    
## Get death flags
cat("> Get death events for test data ... ")
survFlags <- getEventVec(demoTest,0)
cat("Done\n")

################
## IV.i) ROC  ##
################
## Receiver Operating Characteristic (ROC) Figure 
#################################################
cat("> Drawing ROC graph\n")
pdf("ROC.pdf")
cat("> Calling Receiver Operating Characteristic (ROC) function for chronological age\n")
ROCHR <- plot.roc(survFlags,chrAge, col="blue")   ## PCAgeFullTest is for PCAge (original)  
cat("> Calling Receiver Operating Characteristic (ROC) function for PhenoAge\n")
ROCPH <- lines.roc(survFlags,phenoAge, col="#9933CC") ## Levine phenoAge
cat("> Calling Receiver Operating Characteristic (ROC) function for custom clock\n")
ROCBA <- lines.roc(survFlags,bioAge, col="red") ## New bioAge 
legend("bottomright", 
       legend = c("ChronAge","PhenoAge","NewClock"), 
       col = c("blue", "#9933CC", "red"),
       lwd = 2,
       title = "Receiver Operating Characteristic",
       box.lty = 0,
       bg = "NA")
dev.off()
cat("Done\n")

## ROC AUC tests
################
cat("> Doing statistical comparison: Custom clock BA vs PhenoAge  ... ")
testObjCustVsPhen <- roc.test(ROCBA,ROCPH)
cat("Done \n> START ROC STATISTICS >>> \n")
print(testObjCustVsPhen)
cat("<<< END STATS\n\n")


###################
## IV.ii) NOISE  ##
###################
## Investigate effect of gaussian noise on BA estimates  
#######################################################
cat("> 2) Compare impact of stochastic noise on PhenoAge and custom clock BA\n")
cat("--------------------------------------------------------------------------\n")    
## Compare PhenoAge and custom clock in terms of effect of stochastic noise 
cat("> Preparing data - comparing BA with/without addition of 10% gaussian noise to all variables   ... ")

## DATA IMPORT / CONSTRUCTION
## 1) Calculate relative error in BA estimate for phenoAge and custom clock 
bioAgeRelErr <-  (bioAge_stoch-bioAge)/bioAge
## 2) Read in PhenoAge and PhenoAge from data with 10% noise added and calculate relative error
phenoAgeRelErr <- (PA_stoch - PA)/PA
## 3) Add these errors as 1st and 2nd clock to bioAgeRelErr DF
errBA <- cbind(bioAgeRelErr,rep(2,length(bioAgeRelErr)))
colnames(errBA) <- c("value","clock")
errPA <- cbind(phenoAgeRelErr,rep(1,length(phenoAgeRelErr)))
## Remove NA values
colnames(errPA) <- c("value","clock")
errPA <- errPA[!is.na(errPA[,"clock"]),]
noiseDF <- data.frame(rbind(errBA,errPA))
noiseDF$clock <- as.factor(noiseDF$clock)
cat("Done\n")
## 4) Replot figure 2c, comparing error distribution for phenoAge and custom BA clock 
    mFontSize <- 10
    legSize <- 0.8
##  Plot histogram overlay figure
cat("> Drawing histograms   ... ")
histCompClocks <- ggplot() +
    geom_density(data=noiseDF, aes(x=value,  group=clock, fill=clock,),
                              bw=0.02, alpha=0.5, adjust=1, bounds=c(-0.7,0.7)) +
    scale_fill_manual(values=c("orange","#33FFFF"), labels=c("PhenoAge","NewClock")) +
    theme(text = element_text(size = mFontSize)) +
    theme(legend.key.size=unit(legSize,"line"), legend.position=c(0.8,0.9), legend.title=element_blank()) + 
    guides(color = guide_legend(override.aes = list(size = 10))) +
    labs (x = "Relative Error" , y = "Density") +
cat("Done\n")

## Write out overlayed histograms as pdf file
histOut <- "histCompNoise.pdf"
cat("> Writing figure as pdf file: [",histOut,"] ...")
pdf(histOut)
print(histCompClocks)
dev.off()
cat(" Done\n\n\n")


## All done
cat("#################################################################################\n") 
cat("<<< END [",sourceFile,"]\n\n\n")























