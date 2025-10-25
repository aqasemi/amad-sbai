# An Epigenetic Biomarker of Aging for Lifespan and Healthspan

**Authors:** Morgan E. Levine¹, Ake T. Lu¹, Austin Quach¹, Brian H. Chen², Themistocles L. Assimes³, Stefania Bandinelli⁴, Lifang Hou⁵, Andrea A. Baccarelli⁶, James D. Stewart⁷, Yun Li⁸, Eric A. Whitsel⁷⁹, James G. Wilson¹⁰, Alex P. Reiner¹¹, Abraham Aviv¹², Kurt Lohman¹³, Yongmei Liu¹⁴, Luigi Ferrucci²,*, Steve Horvath¹,¹⁵,*

¹Department of Human Genetics, David Geffen School of Medicine, University of California Los Angeles, Los Angeles, CA 90095, USA  
²Longitudinal Studies Section, Translational Gerontology Branch, National Institute on Aging, National Institutes of Health, Baltimore, MD 21224, USA  
³Department of Medicine, Stanford University School of Medicine, Stanford, CA 94305, USA  
⁴Geriatric Unit, Azienda Toscana Centro, Florence, Italy  
⁵Center for Population Epigenetics, Robert H. Lurie Comprehensive Cancer Center and Department of Preventive Medicine, Northwestern University Feinberg School of Medicine, Chicago, IL 60611, USA  
⁶Laboratory of Environmental Epigenetics, Departments of Environmental Health Sciences and Epidemiology, Columbia University Mailman School of Public Health, New York, NY 10032, USA  
⁷Department of Epidemiology, Gillings School of Global Public Health, University of North Carolina, Chapel Hill, NC 27599, USA  
⁸Department of Genetics, Department of Biostatistics, Department of Computer Science, University of North Carolina, Chapel Hill, NC 27599, USA  
⁹Department of Medicine, School of Medicine, University of North Carolina, Chapel Hill, NC 27599, USA  
¹⁰Department of Physiology and Biophysics, University of Mississippi Medical Center, Jackson, MS 39216, USA  
¹¹Public Health Sciences Division, Fred Hutchinson Cancer Research Center, Seattle, WA 98109, USA  
¹²Center of Human Development and Aging, New Jersey Medical School, Rutgers State University of New Jersey, Newark, NJ 07103, USA  
¹³Department of Biostatistics, Division of Public Health Sciences, Wake Forest School of Medicine, Winston-Salem, NC 27157, USA  
¹⁴Department of Epidemiology & Prevention, Division of Public Health Sciences, Wake Forest School of Medicine, Winston-Salem, NC 27157, USA  
¹⁵Department of Biostatistics, Fielding School of Public Health, University of California Los Angeles, Los Angeles, CA 90095, USA  

*Co-senior authors  

**Correspondence to:** Steve Horvath; email: shorvath@mednet.ucla.edu  

**Keywords:** epigenetic clock, DNA methylation, biomarker, healthspan  

**Received:** March 20, 2018  
**Accepted:** April 8, 2018  
**Published:** April 17, 2018  

**Copyright:** Levine et al. This is an open-access article distributed under the terms of the Creative Commons Attribution License (CC BY 3.0), which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

## Abstract

Identifying reliable biomarkers of aging is a major goal in geroscience. While the first generation of epigenetic biomarkers of aging were developed using chronological age as a surrogate for biological age, we hypothesized that incorporation of composite clinical measures of phenotypic age that capture differences in lifespan and healthspan may identify novel CpGs and facilitate the development of a more powerful epigenetic biomarker of aging. Using an innovative two-step process, we develop a new epigenetic biomarker of aging, DNAm PhenoAge that strongly outperforms previous measures in regards to predictions for a variety of aging outcomes, including all-cause mortality, cancers, healthspan, physical functioning, and Alzheimer's disease. While this biomarker was developed using data from whole blood, it correlates strongly with age in every tissue and cell tested. Based on an in-depth transcriptional analysis in sorted cells, we find that increased epigenetic, relative to chronological age, is associated with increased activation of pro-inflammatory and interferon pathways, and decreased activation of transcriptional/translational machinery, DNA damage response, and mitochondrial signatures. Overall, this single epigenetic biomarker of aging is able to capture risks for an array of diverse outcomes across multiple tissues and cells, and provide insight into important pathways in aging.

## Introduction

One of the major goals of geroscience research is to define ‘biomarkers of aging’ [1, 2], which can be thought of as individual-level measures of aging that capture inter-individual differences in the timing of disease onset, functional decline, and death over the life course. While chronological age is arguably the strongest risk factor for aging-related death and disease, it is important to distinguish chronological time from biological aging. Individuals of the same chronological age may exhibit greatly different susceptibilities to age-related diseases and death, which is likely reflective of differences in their underlying biological aging processes. Such biomarkers of aging will be crucial to enable evaluation of interventions aimed at promoting healthier aging, by providing a measurable outcome, which unlike incidence of death and/or disease, does not require extremely long follow-up observation.

One potential biomarker that has gained significant interest in recent years is DNA methylation (DNAm). Chronological time has been shown to elicit predictable hypo- and hyper-methylation changes at many regions across the genome [3-7], and as a result, the first generation of DNAm based biomarkers of aging were developed to predict chronological age [8-13]. The blood-based algorithm by Hannum [10] and the multi-tissue algorithm by Horvath [11] produce age estimates (DNAm age) that correlate with chronological age well above r=0.90 for full age range samples. Nevertheless, while the current epigenetic age estimators exhibit statistically significant associations with many age-related diseases and conditions [14-27], the effect sizes are typically small to moderate. One explanation is that using chronological age as the reference, by definition, may exclude CpGs whose methylation patterns don’t display strong time-dependent changes, but instead signal the departure of biological age from chronological age. Thus, it is important to not only capture CpGs that display changes with chronological time, but also those that account for differences in risk and physiological status among individuals of the same chronological age.

Previous work by us and others have shown that “phenotypic aging measures”, derived from clinical biomarkers [28-32], strongly predict differences in the risk of all-cause mortality, cause-specific mortality, physical functioning, cognitive performance measures, and facial aging among same-aged individuals. What’s more, in representative population data, some of these measures have been shown to be better indicators of remaining life expectancy than chronological age [28], suggesting that they may be approximating individual-level differences in biological aging rates. As a result, we hypothesize that a more powerful epigenetic biomarker of aging could be developed by replacing prediction of chronological age with prediction of a surrogate measure of "phenotypic age" that, in and of itself, differentiates morbidity and mortality risk among same-age individuals.

## Overview of the Statistical Model and Analysis

Our development of the new epigenetic biomarker of aging proceeded along three main steps (Fig. 1). In step 1, a novel measure of ‘phenotypic age’ was developed using clinical data from the third National Health and Nutrition Examination Survey (NHANES). Details on the phenotypic age estimator can be found in Table 1 and in Supplement 1. In step 2, DNAm from whole blood was used to predict phenotypic age, such that:

```
DNAm PhenoAge = intercept + CpG₁ × β₁ + CpG₂ × β₂ + ... + CpG₅₁₃ × β₅₁₃
```

The coefficient values of this model can be found in Supplement 2 (Table S6). Predicted estimates from this model represent a person’s epigenetic age, which we refer to as ‘DNAm PhenoAge’. Using multiple independent datasets, we then tested whether DNAm PhenoAge was associated with a number of aging-related outcomes. We also tested whether it differed as a function of social, behavioral, and demographic characteristics, and whether it was applicable to tissues other than whole blood.

Finally, in step 3, we examine the underlying biology of the 513 CpGs in the DNAm PhenoAge measure by examining differential expression, GO and pathway enrichment, chromosomal locations, and heritability.

### Figure 1: Roadmap for Developing DNAm PhenoAge

The roadmap depicts our analytical procedures. In step 1, we developed an estimate of ‘Phenotypic Age’ based on clinical measures. Phenotypic age was developed using the NHANES III as training data, in which we employed a proportional hazard penalized regression model to narrow 42 biomarkers to 9 biomarkers and chronological age. This measure was then validated in NHANES IV and shown to be a strong predictor of both morbidity and mortality risk. In step 2, we developed an epigenetic biomarker of phenotypic age, which we call DNAm PhenoAge, by regressing phenotypic age (from step 1) on blood DNA methylation data, using the InCHIANTI data. This produced an estimate of DNAm PhenoAge based on 513 CpGs. We then validated our new epigenetic biomarker of aging, DNAm PhenoAge, using multiple cohorts, aging-related outcomes, and tissues/cells. In step 3, we examined the underlying biology of the 513 CpGs and the composite DNAm PhenoAge measure, using a variety of complementary data (gene expression, blood cell counts) and various genome annotation tools including chromatin state analysis and gene ontology enrichment.

## Methodology Focus: Estimating Phenotypic Age from Clinical Biomarkers

### Dataset Handling and Cleaning (Step 1)
- **Training Dataset:** NHANES III (nationally-representative sample of US adults, n=9,926 with complete biomarker data; over 23 years of mortality follow-up).
  - Inclusion: Adults with complete data on 42 clinical biomarkers (selected based on availability in both NHANES III and IV).
  - Exclusion: Incomplete biomarker data (resulting in analytical sample of n=9,926).
  - Biomarkers Considered: Represented physiological systems (e.g., liver, kidney, metabolic, inflammation, immune; full list in Supplement 1).
- **Model Development:** Cox penalized regression model (proportional hazards) where hazard of mortality was regressed on 42 biomarkers + chronological age.
  - Variable Selection: 10-fold cross-validation to select 10 variables (9 biomarkers + age) based on minimizing error.
  - Final Phenotypic Age Formula: Linear combination of selected variables (in years; details in Methods and Supplement 1).
- **Validation Dataset:** NHANES IV (n=6,209 nationally-representative US adults; up to 12 years mortality follow-up).
  - No additional cleaning mentioned beyond complete data; correlations and HRs computed adjusting for chronological age.
- **Outcomes Assessed:** All-cause mortality, cause-specific mortality (competing risks), comorbidity count, physical functioning (Supplement 1: Fig. S1).
- **Key Results:** Phenotypic age correlates with chronological age at r=0.94; predicts mortality (e.g., HR=1.09 per year increase, p=3.8E-49).

#### Table 1: Phenotypic Aging Measures and Gompertz Coefficients

| Variable                  | System       | Units          | Weight   |
|---------------------------|--------------|----------------|----------|
| Albumin                   | Liver        | g/L            | -0.0336 |
| Creatinine                | Kidney       | umol/L         | 0.0095  |
| Glucose, serum            | Metabolic    | mmol/L         | 0.1953  |
| C-reactive protein (log)  | Inflammation | mg/dL          | 0.0954  |
| Lymphocyte percent        | Immune       | %              | -0.0120 |
| Mean (red) cell volume    | Immune       | fL             | 0.0268  |
| Red cell distribution width | Immune    | %              | 0.3306  |
| Alkaline phosphatase      | Liver        | U/L            | 0.0019  |
| White blood cell count    | Immune       | 1000 cells/uL  | 0.0554  |
| Age                       |              | Years          | 0.0804  |

#### Table 2: Mortality Validations for Phenotypic Age

| Mortality Cause              | Cases | HR   | P-Value   |
|------------------------------|-------|------|-----------|
| All-Cause                    | 1052  | 1.09 | 3.8E-49  |
| Aging-Related                | 661   | 1.09 | 4.5E-34  |
| CVD                          | 272   | 1.10 | 5.1E-17  |
| Cancer                       | 265   | 1.07 | 7.9E-10  |
| Alzheimer's                  | 30    | 1.04 | 2.6E-1   |
| Diabetes                     | 41    | 1.20 | 1.9E-11  |
| Chronic lower respiratory diseases | 53 | 1.09 | 6.3E-4   |

## Methodology Focus: An Epigenetic Biomarker of Aging (DNAm PhenoAge)

### Dataset Handling and Cleaning (Step 2)
- **Training Dataset:** InCHIANTI (n=456 participants at two time points: 1998 and 2007; age range 21-100 years).
  - Inclusion: Complete clinical measures for phenotypic age estimation + DNAm data.
  - DNAm Platform: Illumina 27K, 450K, EPIC arrays; harmonized to 20,169 CpGs available across all three platforms for cross-platform usability.
  - Longitudinal Aspect: Used cross-sectional data for training; tested changes over time (mean ΔDNAm PhenoAge=8.51 years vs. Δphenotypic age=8.88 years; correlation r=0.74, p=3.2E-80; Supplement 1: Fig. S2).
- **Model Development:** Elastic net regression (10-fold cross-validation) regressing phenotypic age on 20,169 CpGs.
  - Variable Selection: Selected 513 CpGs; linear combination yields DNAm PhenoAge (mean=58.9, SD=18.2, range=9.1-106.1).
  - Coefficients: Available in Supplement 2 (Table S6).
- **Validation Datasets:** Five independent cohorts (total n>8,000):
  - Women's Health Initiative (WHI; n=2,016 + n=2,191; Illumina 450K).
  - Framingham Heart Study (FHS; n=2,553; 450K).
  - Normative Aging Study (NAS; n=657; 450K).
  - Jackson Heart Study (JHS; n=1,747; EPIC).
  - Cleaning: No explicit cleaning details; assumes complete DNAm and outcome data; adjusted for chronological age, race/ethnicity, pack-years where applicable.
- **Outcomes Assessed:** All-cause mortality (Cox PH models), morbidity (e.g., comorbidity count, disease-free status, physical functioning, CHD; logistic/linear regression), smoking stratification (Supplement 1: Table S4, Fig. S4).
- **Comparisons:** To Hannum and Horvath DNAm Age; ROC for 10/20-year mortality (Supplement 1: Table S2-S3).

### Figure 2: Mortality Prediction by DNAm PhenoAge
(A) Forest plot for fixed-effect meta-analysis (Cox PH models, age-adjusted) across five cohorts. DNAm PhenoAge: HR=1.045 per year (p=7.9E-47); outperforms Hannum (p=1.7E-21) and Horvath (p=4.5E-5). (B) Kaplan-Meier survival (WHI Sample 1: fastest vs. slowest agers). (C) Predicted survival at age 50 (fastest: 81 years; average: 83.5 years; slowest: 86 years).

#### Table 3: Morbidity Validation for DNAm PhenoAge (Excerpt; Full in Paper)

| Sample              | Comorbidity Coef (P) | Disease-Free Coef (P) | CHD Risk Coef (P) | Physical Functioning Coef (P) |
|---------------------|---------------------|-----------------------|-------------------|------------------------------|
| **DNAm PhenoAge**  |                     |                       |                   |                              |
| WHI BA23 White     | 0.008 (2.38E-01)   | -0.002 (3.82E-01)    | 0.016 (5.36E-02) | -0.396 (1.04E-04)           |
| ... (other rows)   | ...                 | ...                   | ...               | ...                          |
| Meta P (Stouffer)  | 1.95E-20            | 2.14E-10              | 3.35E-11          | 2.05E-13                     |
| **DNAmAge Hannum** |                     |                       |                   |                              |
| ...                | ...                 | ...                   | ...               | ...                          |
| Meta P (Stouffer)  | 1.50E-08            | 1.64E-04              | 1.40E-05          | 2.03E-05                     |
| **DNAmAge Horvath**|                     |                       |                   |                              |
| ...                | ...                 | ...                   | ...               | ...                          |
| Meta P (Stouffer)  | 3.26E-06            | 6.36E-07              | 1.49E-01          | 1.43E-03                     |

## DNAm PhenoAge and Smoking

- **Handling:** Stratified analyses by smoking status (never/current/former; n=1,097/209/710); adjusted for pack-years.
- **Associations:** DNAm PhenoAge differs by status (p=0.0033; Supplement 1: Fig. S3A); no robust pack-years link (Fig. S3B-D).
- **Stratified Results:** Mortality (smokers: HR=1.050, p=7.9E-31; non-smokers: HR=1.033, p=1.2E-10); similar for morbidity (Table S4).

## DNAm PhenoAge in Other Tissues

- **Datasets:** Multi-tissue validation (e.g., brain r=0.54-0.92; breast r=0.47; buccal r=0.88; overall r=0.71).
- **Handling:** DNAm PhenoAge computed on non-blood tissues; no platform-specific cleaning noted.

### Figure 3: DNAm PhenoAge Correlations in Tissues
Scatterplots of chronological age vs. DNAm PhenoAge across tissues (e.g., brain, breast, buccal, etc.).

## Alzheimer's Disease and Brain Samples

- **Dataset:** ROS/MAP (n≈700 post-mortem dorsolateral prefrontal cortex samples).
- **Handling:** Pathological diagnosis via autopsy; adjusted for age.
- **Associations:** AD cases have +1 year older DNAm PhenoAge (p=4.6E-4); correlates with amyloid (r=0.094, p=0.012), plaques (r=0.11, p=0.0032), tangles (r=0.10, p=0.0073).

### Figure 4: DNAm PhenoAge in DLPFX by AD Status
Boxplot showing higher DNAm PhenoAge in AD cases vs. controls (age-adjusted).

## Reimplementation Notes
- **Step 1 (Phenotypic Age):** Use Cox PH with elastic net penalty (e.g., glmnet in R/Python) on NHANES-like data; 10-fold CV for var selection.
- **Step 2 (DNAm PhenoAge):** Elastic net regression (glmnet) on CpG β-values; harmonize CpGs across arrays; validate longitudinally and cross-cohort.
- **Data Cleaning General:** Focus on complete cases; adjust for age/race/smoking; meta-analyze with fixed-effects (e.g., metafor package).
- **Tools:** R (survival, glmnet) or Python (lifelines, scikit-learn); supplements for full coeffs/formulas. For full biology (Step 3), use GO enrichment tools (e.g., DAVID).