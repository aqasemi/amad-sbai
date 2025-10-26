
## NOTE: Much of this script is written to be instructive rather than fast or elegant.
##       Nothing here needs to run more than once or on very large datasets ...
library(survival)
library(ggplot2)

## Do full cleanup of workspace
rm(list=ls())
cat("\n FULL PCAge script:\n\n 1) Select and clean NHANES data\n 2) Dimensionality reduction by SVD\n 3) Fit COX model and construct clock\n 4) Apply to test cohort\n 5) Draw scatter plot\n\n")

####################################################
############  FUNCTION DEFINITIONS  ################
####################################################

markIncsFromCodeBook<-function(codeBook) {

    ## Selects all columns marked with "1" in codebook for inclusion 
    incFlags<-codeBook[,"Data"]   
    incNames<-as.character(codeBook[,1]) 
    incList<-cbind(incNames,incFlags)
    return(incList)
}


qDataMatGen<-function(masterData,incList) {

    ## Loop over masterData and keep any column that has a zero in the "data" column
    allTerms<-colnames(masterData)
    nTerms<-dim(masterData)[2]   
    nIncFlags<-dim(incList)[1]   
    
    ## The first column of the qDataMatrix does also has to be SEQN number - add these first
    qDataNames<-"SEQN"
    qDataMatrix <- masterData[,1]
    
    ## Loop over all terms (columns) in masterData - extract one term (column) at a time
    for (i in 2:nTerms) {    
        nextTerm<-allTerms[i]
        flag<-0   
        for (j in 1:nIncFlags) {
            if (incList[j,1]==nextTerm) { 
                flag <- incList[j,2]
            }
        }
        if (flag == 0) {
            ## If include == 0, we will include that parameter in the qDataMatrix 
            qDataColumn <- masterData[,i]    
            qDataMatrix<-cbind(qDataMatrix,qDataColumn)  
            qDataNames<-c(qDataNames,nextTerm)  
        }
    }
    colnames(qDataMatrix) <- qDataNames  ## Update all column names  
    return(qDataMatrix)  
}


dropCols<-function(dataSet,incList) {

    ## Simply drops everything not flagged with 1 in incList
    incInds<-which(incList[,2]==1)
    incTerms<-incList[,1][incInds]

    nTerms<-length(incTerms)
    ## The first column is always the subject number (SEQN) - lets add that
    outData<-dataSet[,1]

    for (i in 2:nTerms) {
        nexTerm<-incTerms[i]
        ## Select column in original data with this name
        nextCol<-which(colnames(dataSet)==nexTerm)
        ## Copy the data into vector
        colData<-dataSet[,nextCol]
        ## Make this vector the next column of the output data
        outData<-cbind(outData,colData)
        ## Update column name to be nexTerm
        colnames(outData)[i]<-nexTerm
        
    }
    colnames(outData)[1]<-"SEQN"
    return(outData)
}


nonAccidDeathFlags <-function(qMat) {

    ## Here we will return keep flags for all subjects who die of non-accidental deaths
    ## The cause of death (leading) is recorded (if known) in the questionnaire data matrix
    ## qDatMat in the "UCOD_LEADING" column
    ## The values in "UCOD_LEADING" are:
    ## 001 = Disease of the heart
    ## 002 = Malignant neoplasms
    ## 003 = Chronic lower respiratory disease
    ## 004 = Accidents and unintentional injuries
    ## 005 = Cerebrovascular disease
    ## 007 = Diabetes
    ## 008 = Influenza and pneumonia
    ## 009 = Nephritis, kidney issues
    ## 010 = All other (residuals)
    ## NA  = No info (the vast majority of cases)

    ## Extract cause of deaths
    causeOfDeath <- qMat[,"UCOD_LEADING"]
    ## Then drop NAs (turn into zeros) 
    causeOfDeath[is.na(causeOfDeath)]<-0
    keepFlags <- causeOfDeath!=4
    
    return(keepFlags) 
                
}


selectAgeBracket<-function(qMat,ageCutLower,ageCutUpper) {

    ## Apply an age bracket to dataset - only retain samples within age bracket
    keepRows<-((qMat[,"RIDAGEYR"]>=ageCutLower) & (qMat[,"RIDAGEYR"]<=ageCutUpper))
    return(keepRows)
}



dropNAcolumns<-function(dataSet,pNAcut,incSwitch,verbose) {
        
    ## This takes a single cutoff and drops all columns (features) that contain more NAs than allowed
    ## by the cutoff. However, if incFlag is set (==1), we will also force inclusion of features
    ## flagged in the include column (Include) in the codebook 

    ## If incSwitch is not set, set all include flags to zero (force include nothing)
    nRows<-dim(dataSet)[1]
    nCols<-dim(dataSet)[2]
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
    keepCols <- (naColP < pNAcut)

    ## Finally, recover all columns that we decided to force (retain) 
    keepCols<-keepCols | (forceFlags == 1)
    dataSet<-dataSet[,keepCols]
    varNames<-colnames(dataSet)
    humNames<-varNames
    
    for (i in 2:length(varNames)) {
        varName<-varNames[i]
    }
    return(dataSet)
}


getNonNARows<-function(dataSet) {

    ## Identify rows that contain NAs and drop them 
    keepRows<-(rowSums(is.na(dataSet))==0)
    return(keepRows)
}


popPCFIfs1 <- function(qDataMat) {
        ## This will calculate a frailty index 
      
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

        ## Binary yes/no vec 
        binVec <- cbind((BPQ020==1),((DIQ010==1) | (DIQ010==3)),(KIQ020==1),(MCQ010==1),(MCQ053==1),(MCQ160A==1),(MCQ160C==1),(MCQ160D==1),(MCQ160E==1),(MCQ160F==1),(MCQ160G==1),(MCQ160I==1),(MCQ160J==1),(MCQ160K==1),(MCQ160L==1),(MCQ220==1),(OSQ010A==1),(OSQ010B==1),(OSQ010C==1),(OSQ060==1),(PFQ056==1),(HUQ070==1))

    sumOverBinVec <- rowSums(binVec)/22 
    return(sumOverBinVec)
}

popPCFIfs2 <- function(qDataMat) {

    ## Self reported health 
    
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

    ## Uses HUQ050 - number of visits to hospital. Set comment codes to 0.
    HUQ050  <- qDataMat[,"HUQ050"]
    HUQ050[is.na(HUQ050)] <- 0
    HUQ050[(HUQ050==77)] <- 0
    HUQ050[(HUQ050==99)] <- 0
    return(HUQ050)
}


normAsZscores<-function(dataSet) {

    ## Normalize data by column average for each column independently
    ## Calculate column averages for all features other than sequence numbers
    nRows<-dim(dataSet)[1]
    nCols<-dim(dataSet)[2]

    ##  Calculate means and SDs for each column (feature) 
    colAvs<-colMeans(dataSet)
    colSDs<-apply(dataSet,2,sd)
    
    ## Make normalized matrix by turning each value into z-score
    dataMatN<-matrix(0,nRows,nCols)

    ## There is certainly a more elegant matrix way of doing this - but lets just do a
    ## simple loop - column 1 is still the subject seq number - will not be normalized
    dataMatN[,1]<-dataSet[,1]
    colnames(dataMatN)<-colnames(dataSet)

    ## Loop over all columns - starting from column 2 (first data col) 
    for (col in 2:nCols) {
        mean<-colAvs[col]
        sd<-colSDs[col]
        for (row in 1:nRows) {
            zScore<-(dataSet[row,col]-mean)/sd
        dataMatN[row,col]<-zScore
        }
    }
    return(dataMatN)
}


getEventVec <- function (qMatrix, cause) {

    ## Read qDataMatrix and determine if individual died during study period or was censored
    ## that is, survived beyond the end of the study 

    if (cause == 0) {  ## IF cause is 0, we do not care what ppl died from and report all death 
        eventFlags<-qMatrix[,"MORTSTAT"] 
        return (eventFlags)
    }
    if (cause !=0) {  ## If cause is > 0, we will report only specific causes of death
        eventFlags<-qMatrix[,"MORTSTAT"]
        CODFlags <- qMatrix[,"UCOD_LEADING"]

        ## The values in "UCOD_LEADING" are:
        ## 001 = Disease of the heart
        ## 002 = Malignant neoplasms
        ## 003 = Chronic lower respiratory disease
        ## 004 = Accidents and unintentional injuries
        ## 005 = Cerebrovascular disease
        ## 007 = Diabetes
        ## 008 = Influenza and pneumonia
        ## 009 = Nephritis, kidney issues
        ## 010 = All other (residuals)
        ## NA  = No info (the vast majority of cases)
        
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
        if (cause == 7) { ## Deaths directly from diabetes only 
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
    }
}


getSurvTime <- function (qMatrix) {

    ## Function returns time between enrollment and end of follow up
    ## NOTE: THIS IS REALLY NOT SURVIVAL TIME BUT TIME TO FOLLOW UP - contingent on eventFlags ... 
    survTimes<-qMatrix[,"PERMTH_EXM"]
    return (survTimes)

    }

getDataDims <- function (dataMatrix) {

    ## Dump dimensions of data matrix    
    cat(paste("       [Data matrix now has ",dim(dataMat)[1]," rows (subjects) and ",dim(dataMat)[2],"columns (features) ] \n\n"))

}

makeSurvObject <- function (qMatrix,causeOfDeath) {

    ## Take survival times and censor vector (from demographic matrix) and make a survival object 
    ## First, get event flags from qMatrix - subjects that died have 1, those that survived have 0 in here
    eventFlags <- getEventVec(qMatrix,causeOfDeath)
    ## Then, get survival times (either time between exam and death (if dead) or time to end of follow up (alive)
    times <- getSurvTime(qMatrix)

    survObj<-Surv(times,eventFlags) 
    return(survObj) 
    
}

calcBioAge <- function (coxModelNew,nullModel,dataTable) {
   
    ## This will take the coxModel plus the input data table (covariates used for the cox model)
    ## it will then loop over the data table and calculate the delta ages for each individual    

    ## The cox model assumes that the hazard function hi(t) for each subject i
    ## can be broken down into log additive terms according to the linear (lm) model
    ## plus the universal time-dependent (lengths of follow up) term, h0(t)             
    ## e.g. here: hi(t) = h0(t)*exp(beta1*startAge + beta2*x2 + beta3*x3 + beta4*x4 ... )
    betasCOX<-coxModelNew$coefficients
    betasNull <- nullModel$coefficients
    
    ## Assume that h is the mortality function according to gompertz - we can infer that:                         
    ##     beta1*ageStart == ln(2)/MRDT*ageStart
    ## <=> beta1 == ln(2)/MRDT
    ## <=> MRDT == ln(2)/beta1
    ##
    betaOne <- betasCOX[1]
    MRDTfit <- round(log(2)/betaOne,2)
    
    ## Using this, we can convert the risk ratios from the full model into age differences:
    ## hi(t) = h0(t)*exp(ln(2)/MRDT*ageStart+beta2*x2 + beta3*x3 + ....)
    ## this means we can define a biological "delAge"
    ## via: delAge * ln(2)/MRDT == beta2*x2 + beta3*x3 + beta4*x4
    
    ## USE BUILD IN PREDICT FUNCTION
    riskMod <- predict(coxModelNew,newdata=dataTable,type="risk")
    riskNull <- predict(nullModel,newdata=dataTable,type="risk")
    
    logRiskRatio <- log(riskMod/riskNull) 
    ageBioDelta <- logRiskRatio/log(2)*MRDTfit   
    return(ageBioDelta)           
}

### <<< END FUNCTION DEFS



#########################################################
###     >>>        MAIN PCAGE SCRIPT       <<<        ###
#########################################################
cat("\n\n >>> Starting main script <<< \n")    

###################################
## I) DATA FILES AND DATA IMPORT ##
###################################
cat(" > I   Reading files ...")    
## First, read in csv of NHANES IV (continuous) data 
## These data were generated using nhanesA.
dataFile <- "nhanesMerged.csv"
masterData<-read.csv(dataFile)
cat(paste(dataFile,"... ")) 
## Extract variable names
allNames<-colnames(masterData) 
    
## Read the codebook file - this defines the variables we will use
logBookFile <- "codeBook.csv"
codeBook<-read.csv(logBookFile)
cat(paste(logBookFile,"... Done\n\n")) 


##########################
## II) CLEANING UP DATA ##
##########################
cat(" > II  Data cleanup and normalization\n")    
###############################
## II.i) FILTER INPUT MATRIX ##
###############################
## Drop all duplicate columns and non-data columns from master data
## -> This is based on include flags in codebook
cat("     > Selecting features to include based on codebook ...\n")
incList<-markIncsFromCodeBook(codeBook)             
dataMat<-dropCols(masterData,incList)  ## Main data matrix for clock
## First make a questionnaire Data matrix - everything OTHER than the numerical / clinical data
## qDataMat will include anything that is NOT flagged as "data" in the codebook     
qDataMat<- qDataMatGen(masterData,incList)   ## NOTE: This is pretty much the same as dropCols for dataMatrix ...
getDataDims(dataMat)

## Need to drop all subjects for whom we have no information on age
cat("     > Removing subjects with missing age data ...\n")
subSansAge <- which(is.na(qDataMat[,"RIDAGEEX"]))
dataMat<-dataMat[-subSansAge,]
qDataMat <-qDataMat[-subSansAge,]
getDataDims(dataMat)

## Drop all accidental death cases
cat("     > Removing accidental deaths  ...\n")
keepRows <- nonAccidDeathFlags(qDataMat) 
dataMat<-dataMat[keepRows,]
qDataMat<-qDataMat[keepRows,]
getDataDims(dataMat)

## Apply age cutoffs 
## First, remove children (below the age of ageCut) as we do not want to deal with
## development but aging - also remove individuals over 84 as the age data is top
## coded at 85 (e.g. 100 is recorded as 85)
ageLower<-40
ageUpper<-84   
cat(paste("     > Applying age cutoff - include only ages [",ageLower,",",ageUpper,"] \n"))
keepRows<-selectAgeBracket(qDataMat,ageLower,ageUpper)
dataMat<-dataMat[keepRows,]
getDataDims(dataMat)
## NOTE: Anytime that we drop rows (subject), we have to also drop the same rows
## from demographic data and update the sequence data:
qDataMat<-qDataMat[keepRows,]

## Drop columns (variables) that are missing from pNAcut*100% of subjects from dataset
pNAcut<-0.09
cat(paste("     > Doing NA remove with per-column NA cutoff of",pNAcut*100,"% \n"))
dataMat<-dropNAcolumns(dataMat,pNAcut,1,0)
getDataDims(dataMat)

## Now that we have removed sparse columns, and comment codes, remove subjects (rows)
## that still have NAs
cat("     > Drop all subjects that still contain NA values ...\n")
keepRows<-getNonNARows(dataMat) 
dataMat<-dataMat[keepRows,]
## Also need to again update the demographic matrix to remove the same ppl
qDataMat<-qDataMat[keepRows,]
getDataDims(dataMat)


###############################
## Populate derived features ##     
###############################
cat("     > Populate  derived features  ...")
fs1Score <- popPCFIfs1(qDataMat)
fs2Score <- popPCFIfs2(qDataMat) 
fs3Score <- popPCFIfs3(qDataMat)
dataMat<-cbind(dataMat,fs1Score,fs2Score,fs3Score)
cat(" Done\n\n")


##########################################
## III) DATA NORMALIZATION AND SCALING  ##
##########################################
cat(" > III  Applying normalization by z-score ...")
dataMatNorm<-normAsZscores(dataMat) 
cat(" Done\n\n")
####################################
## IV) DIMENSIONALITY REDUCTION   ##
####################################

#######################################
## IV.i) Do the SVD of training set  ##
#######################################
cat(" > IV  SVD for PCA and coordinate transformation\n")
## The input matrix needs to be centered and scaled for SVD to be
## equivalent to PCA
cat("     > Calculating SVD for training set  ....\n")
## Do the SVD ONLY for the 99/00 cohort    
inputMat <- dataMatNorm[,-1] ## Drop SEQn (patient number) from SVD input matrix
inputMat99 <- inputMat[(qDataMat[,"yearsNHANES"]==9900),]  ## Only use 99/00 rows - all columns
inputMat01 <- inputMat[(qDataMat[,"yearsNHANES"]==102),]  ## Only use 01/02 rows - all columns
svdData99<-svd(inputMat99)    

## extract diagonal matirx of singular values as well as
## right (V) and left (U) singular vectors
diagDat99<-diag(svdData99$d)
uMatDat99<-svdData99$u
vMatDat99<-svdData99$v

## uMat (left singular vector) is of dimension nrOfSamples x nrOfSVDs
mSamples99<-dim(uMatDat99)[1]
nSVs99<-dim(uMatDat99)[2]  ## Number of PCs / SVD 

## Now we need to construct the SVD coordinates for each subject in BOTH waves (99/00 and 01/02)
## This is simply by projecting samples back into SVD coordinates. The SVD provides an orthonormal 
## base of feature space - so this is simply a question of taking a feature vector and multiplying
## it by each PC load vector 
cat("     > Project testing data into PC coordinates from training set ...")
## First get coordinates of subjects from 99 wave in SVD coordinates 
pcMat99 <- uMatDat99 %*% diagDat99

## Then transform the 01/02 wave subject coordinates into SVD coordinates
## Simple projection based on the base for the 99/00 cohort
mSamples01 <- dim(inputMat01)[1]
pcMat01<-matrix(0,mSamples01,nSVs99)

## Doing loop to calculate coordinates for samples in terms of PCs
## NOTE: This is more instructive than efficient or elegant - you are welcome ...
for (sample in 1:mSamples01) {
    ## Current sample is current row of data (inputMat01) matrix
    curSample<-inputMat01[sample,]
    ## Now loop over all nSVs and determine
    for (pcNr in 1:nSVs99) {
        ## current PC vector is the column
        curPC<-vMatDat99[,pcNr]
        coord<-curSample %*% curPC
                pcMat01[sample,pcNr]<-coord
    }
}

## Merge both the 99/00 and 01/02 waves into one matrix
pcDatMat <- rbind(pcMat99,pcMat01)
cat(" Done\n\n")

##############################################
## IV.iii) Actual dimensionality reduction ##
##############################################
## Scree plot
## The variance explained by the nth SVs is singVal(n)^2/sumOverAllN(singVals^2) - this is also called scree
scree<-svdData99$d^2/sum(svdData99$d^2)
plot(scree, type = "b", pch = 16, xlab = "principal components", ylab = "variance explained")

cat(" > V   Reduce dimensions by dropping PCs ... \n")
## scree[n] * 100 is the percent explaiend by the nth singular vector - lets use this to cut off
## at the point where the nth SV explains less than svCutP % of all variance - that is, where:
## scree[n] becomes less than svCutP/100
svCutP<-1   # % cutoff - that is, the cutoff factor is svCutP/100
cat(paste("     > Applying PC cutoff: (PCs explaining less than ",svCutP,"% of variance are dropped) ... "))
svCut<-which(scree<svCutP/100)

## We will truncate the dataMatrix at this point - dropping all higher SVs / PCs
pcDatMat<-pcDatMat[,1:(svCut[1]-1)]
nSVs <- dim(pcDatMat)[2]

## Name columns by PCNr
colnames(pcDatMat) <- paste("PC",1:nSVs,sep="")
cat(" Done\n\n")
##########################
## V Clock construction ##
##########################
cat(" > VI  Building clock \n")

##################################################################################
## V.1 COX model / clock construction - preparing training and testing datasets ##
##################################################################################
## We will use the 99/00 wave as training set and 01/02 as testing set
cat("     > Splitting data in test and training set based on year of recruitment\n")
trainSam <- (qDataMat[,"yearsNHANES"]==9900)
testSam <- (qDataMat[,"yearsNHANES"]==102)

######################################################
## Extract PC coordinates for subjects (TRAIN/TEST) ##
######################################################
cat("     > Extracting covariates for COX model ... Include:")
## Split the PCA matrix into test and train matrices
cat(paste("First",nSVs,"PCs ..."))
xTrainPCA<-pcDatMat[trainSam,]  
xTestPCA<-pcDatMat[testSam,]

#########################################################
## Extract chron age and sex for subjects (TRAIN/TEST) ##
#########################################################
cat("Chronological age ...")
ageAtExTrain<-qDataMat[trainSam,"RIDAGEEX"]
ageAtExTest<-qDataMat[testSam,"RIDAGEEX"]
cat("Sex ... Done \n")
sexTrain <- qDataMat[trainSam,"RIAGENDR"]
sexTest <- qDataMat[testSam,"RIAGENDR"]

##########################################################################################
## V.2 COX model / clock construction - Assemble covariate matrix for COX  (TRAIN/TEST) ##
##########################################################################################
## Make COX covariate matrix for training set                           
coxCovsTrain <- cbind(ageAtExTrain,xTrainPCA,sexTrain)
colnames(coxCovsTrain)[1]<-"chronAge"  ## The variable names need to be fixed for the cox model function 
colnames(coxCovsTrain)[(nSVs+2)]<-"sex"
coxCovsTrain<-as.data.frame(coxCovsTrain)

## Make COX covariate matrix for test set                           
coxCovsTest <- cbind(ageAtExTest,xTestPCA,sexTest)
colnames(coxCovsTest)[1]<-"chronAge"  ## The variable names need to be fixed for the cox model function 
colnames(coxCovsTest)[(nSVs+2)]<-"sex"
coxCovsTest<-as.data.frame(coxCovsTest)

## Finally, split these into male and female sets
coxCovsTestM <- coxCovsTest[coxCovsTest$sex==1,]
coxCovsTestF <- coxCovsTest[coxCovsTest$sex==2,]
coxCovsTrainM <- coxCovsTrain[coxCovsTrain$sex==1,]
coxCovsTrainF <- coxCovsTrain[coxCovsTrain$sex==2,]

###########################################################
##  V.3 Construct mortality/survival object (TRAIN/TEST) ##
###########################################################
cat("     > Making survival object for COX model  ... \n")
## Extract all demographic data
demoTest<-qDataMat[testSam,]
demoTrain<-qDataMat[trainSam,]

## Make survival objects for survival package 
## Males
survObjTrainM <- makeSurvObject(demoTrain,0)[(demoTrain[,"RIAGENDR"]==1)] 
survObjTestM <- makeSurvObject(demoTest,0)[(demoTest[,"RIAGENDR"]==1)]

## Females
survObjTrainF <- makeSurvObject(demoTrain,0)[(demoTrain[,"RIAGENDR"]==2)] 
survObjTestF <- makeSurvObject(demoTest,0)[(demoTest[,"RIAGENDR"]==2)]

##################################################################################
##  V.4 Make COX models to predict survival from PC coordiantes of training set ##
##################################################################################
cat("     > Fitting COX model  ... \n")
## Null model only depends on chron age - basic gompertz
## Males
nullModelM <- coxph(survObjTrainM ~ chronAge, data=coxCovsTrainM)
## Females
nullModelF <- coxph(survObjTrainF ~ chronAge, data=coxCovsTrainF)

## Now fit model taking into consideration location in feature space (PCs)
## Males
coxModelM <- coxph(survObjTrainM ~ chronAge + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18,data=coxCovsTrainM)
## Females
coxModelF <- coxph(survObjTrainF ~ chronAge + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18,data=coxCovsTrainF)

#######################################################################
##  V.5 Convert risk ratios into age deltas using Gompertz function ##
#######################################################################
cat("     > Converting risk ratios into biological age deltas  ... ")
## Males
chronAgeTrainM <- coxCovsTrainM[,"chronAge"]/12
deltaBioAgeTrainM <- calcBioAge(coxModelM,nullModelM,coxCovsTrainM)/12
bioAgeTrainM <- chronAgeTrainM + deltaBioAgeTrainM/12 
chronAgeTestM <- coxCovsTestM[,"chronAge"]/12
deltaBioAgeTestM <- calcBioAge(coxModelM,nullModelM,coxCovsTestM)/12 
bioAgeTestM <- chronAgeTestM + deltaBioAgeTestM

## Females
chronAgeTrainF <- coxCovsTrainF[,"chronAge"]/12
deltaBioAgeTrainF <- calcBioAge(coxModelF,nullModelF,coxCovsTrainF)/12
bioAgeTrainM <- chronAgeTrainF + deltaBioAgeTrainF/12
chronAgeTestF <- coxCovsTestF[,"chronAge"]/12
deltaBioAgeTestF <- calcBioAge(coxModelF,nullModelF,coxCovsTestF)/12 
bioAgeTestF <- chronAgeTestF + deltaBioAgeTestF
cat(" Done\n")



######################################################### 
###     >>>        MAIN PCAGE SCRIPT       <<<        ###
#########################################################    
cat("\n\n >>> Making Figure <<< \n")   

## Set font sizes here!!
mFontSize <- 9
## change font settings for survfit plots here!!
masterFont <- c(9,"plain","black")

## Legend size
legSize <- 0.5
lineSize <- 0.5
## Set this to 

cat(" > Making scatter plot - chronological age vs biological age \n")
## Use ggplot for pretty graphs - make dataframe with chronAges and bioAges and ageDeltas
df1M  <- data.frame(chronAgeTestM,bioAgeTestM,deltaBioAgeTestM)
df1F <- data.frame(chronAgeTestF,bioAgeTestF,deltaBioAgeTestF)

## df1M contains chronological age, PCAge and delta age for all males in test set (NHANES IV 01/02 wave)
## df1F contains chronological age, PCAge and delta age for all females in test set (NHANES IV 01/02 wave)

   scatPlotBoth <- ggplot() +
        geom_point(data=df1F,
                   aes(x=chronAgeTestF,y=bioAgeTestF),
                   shape=21,
                   fill="#FF6699",
                   color="black", 
                   size=0.8,
                   stroke=0.3) +

        geom_point(data=df1M,
               aes(x=chronAgeTestM,y=bioAgeTestM),
               shape=21,
               fill="#6666FF",
               color="black", 
               size=0.8,
               stroke=0.3) +

   geom_smooth(data = df1M,
                color="#9999FF",
                aes(x=chronAgeTestM,y=bioAgeTestM),
                method=lm,
                fill="gray",
                size = 0.5,
               level=0.99,
               alpha=0.2,
                se=TRUE) +

    geom_smooth(data = df1F,
                color="red",
                aes(x=chronAgeTestF,y=bioAgeTestF),
                method=lm,
                fill="gray",
                alpha=0.5,
                size=0.5,
                level=0.99,
                se=FALSE) +
    theme(text = element_text(size = mFontSize)) +

    labs(x = "Chronological Age (years)",
         y = "PCAge (years)") 

cat(" > Plot: scatPlotBoth - done\n\n")
