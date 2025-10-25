# A Detailed Breakdown of How PhenoAge Was Derived

The **Levine Phenotypic Age ("PhenoAge")** is a *"second-generation"* biological age clock, published in **2018**. Its development was a crucial shift in the field, moving away from *"first-generation"* clocks (like Horvath's) which were trained to predict chronological age.

The researchers understood that predicting chronological age is flawed — you end up finding biomarkers that just track time, not risk. The entire goal of PhenoAge was to create a clock that predicted **mortality risk**.

The derivation was an innovative **two-step process** using the **NHANES III dataset** (which has both clinical biomarkers and long-term mortality follow-up).

---

## Step 1: Creating the "Ground Truth" (The Phenotypic Age Score)

The primary problem was: how do you get a *"ground truth"* for biological age?
The researchers decided to create one based on **mortality**.

### The Goal

To create a single, continuous score for each person that represented their **risk of death**, based on a comprehensive panel of biomarkers.

### The Data

They started with **42 different clinical biomarkers** (plus chronological age) from the NHANES III participants.

### The Model

They used a **Cox proportional hazards model** (a survival analysis model, as detailed in your research text).

* **Target (Y):** All-cause mortality (i.e., who actually died and how long it took)
* **Features (X):** The 42 biomarkers and chronological age

### The "Ground Truth" Output

This model produced a **mortality risk score** for each person.
This score was a powerful predictor of death. The researchers then **scaled this risk score into the intuitive units of "years"**, and this new, calculated score was named **"Phenotypic Age."**

This *Phenotypic Age* score — a complex, 42-biomarker-derived mortality risk score — became the official *"ground truth"* for the next step.

---

## Step 2: Creating the Simple, Usable Clock (The 9-Biomarker Formula)

The 42-biomarker model was powerful, but not practical for a doctor's office. The goal of Step 2 was to create a **simple, cheap proxy model** that could accurately estimate the *Phenotypic Age* score from Step 1.

### The Goal

To create a simple formula that could predict the *Phenotypic Age* score.

### The Data

They used the same **NHANES III participants**.

### The Model

They trained a new machine learning model (**a penalized regression**).

* **Target (Y):** The *Phenotypic Age* score they had just created in Step 1
* **Features (X):** The same 42 biomarkers and chronological age

### The Final Formula

The regression model automatically selected the smallest, most predictive subset of features needed to estimate the *Phenotypic Age* score.
It shrunk the list from **42 biomarkers down to just 9 biomarkers plus chronological age**.

The published formula for **PhenoAge** is the equation for this second model.
It is a **simple, 9-biomarker model** trained to predict the score of a complex, 42-biomarker mortality model.




### Lab tests limitaitons

============================================================
Tests sorted by number of users (ascending):
============================================================
  Glucose: 110251 users
  LYM: 158508 users
  RDW: 166357 users
  MCV: 228310 users
  Albumin: 491331 users
  WBC: 637584 users
  ALP: 638652 users
  Creatinine: 1633341 users


============================================================
Combinations of 3 tests:
============================================================
 1. 209757 users: [Albumin, Creatinine, ALP]
 2. 112089 users: [Creatinine, ALP, WBC]
 3.  83543 users: [Albumin, Creatinine, WBC]
 4.  75431 users: [Albumin, ALP, WBC]
 5.  46978 users: [Creatinine, LYM, WBC]
 6.  36234 users: [Creatinine, MCV, ALP]
 7.  30564 users: [Creatinine, LYM, ALP]
 8.  29990 users: [Creatinine, RDW, ALP]
 9.  25955 users: [Albumin, Creatinine, MCV]
10.  22946 users: [Albumin, Creatinine, LYM]


============================================================
Combinations of 4 tests:
============================================================
 1.  52168 users: [Albumin, Creatinine, ALP, WBC]
 2.  16068 users: [Creatinine, LYM, ALP, WBC]
 3.  15604 users: [Albumin, Creatinine, MCV, ALP]
 4.  14497 users: [Albumin, Creatinine, LYM, ALP]
 5.  13333 users: [Albumin, Creatinine, RDW, ALP]
 6.  12132 users: [Albumin, Creatinine, LYM, WBC]
 7.  11056 users: [Albumin, LYM, ALP, WBC]
 8.   8234 users: [Albumin, Creatinine, Glucose, ALP]
 9.   4181 users: [Creatinine, Glucose, ALP, WBC]
10.   4142 users: [Albumin, Creatinine, Glucose, WBC]



============================================================
Combinations of 5 tests:
============================================================
 1.   7691 users: [Albumin, Creatinine, LYM, ALP, WBC]
 2.   2382 users: [Albumin, Creatinine, Glucose, ALP, WBC]
 3.    979 users: [Albumin, Creatinine, MCV, ALP, WBC]
 4.    898 users: [Albumin, Creatinine, RDW, ALP, WBC]
 5.    653 users: [Albumin, Creatinine, Glucose, MCV, ALP]
 6.    627 users: [Albumin, Creatinine, Glucose, LYM, WBC]
