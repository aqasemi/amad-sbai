import pandas as pd
from pathlib import Path


def create_reference_ranges():
    ranges_data = [
        {
            "BiomarkerID": 1,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 3.5,
            "MaxVal": 5.0,
        },  # Albumin g/dL
        {
            "BiomarkerID": 2,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 44,
            "MaxVal": 147,
        },  # ALP U/L
        {
            "BiomarkerID": 3,
            "RangeType": "clinical",
            "Sex": "M",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 0.74,
            "MaxVal": 1.35,
        },  # Creatinine mg/dL Male
        {
            "BiomarkerID": 3,
            "RangeType": "clinical",
            "Sex": "F",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 0.59,
            "MaxVal": 1.04,
        },  # Creatinine mg/dL Female
        {
            "BiomarkerID": 4,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 74,
            "MaxVal": 106,
        },  # Glucose mg/dL
        {
            "BiomarkerID": 5,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 0.0,
            "MaxVal": 3.0,
        },  # CRP mg/L
        {
            "BiomarkerID": 6,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 3.5,
            "MaxVal": 10.5,
        },  # WBC 10³ cells/µL
        {
            "BiomarkerID": 7,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 20,
            "MaxVal": 45,
        },  # Lymphocyte %
        {
            "BiomarkerID": 8,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 80,
            "MaxVal": 100,
        },  # MCV fL
        {
            "BiomarkerID": 9,
            "RangeType": "clinical",
            "Sex": "All",
            "AgeMin": 18,
            "AgeMax": 120,
            "MinVal": 11.5,
            "MaxVal": 14.5,
        },  # RDW %
        # Longevity ranges (tighter, based on healthy young adults 20-30)
        {
            "BiomarkerID": 1,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 4.0,
            "MaxVal": 4.8,
        },  # Albumin g/dL
        {
            "BiomarkerID": 2,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 60,
            "MaxVal": 120,
        },  # ALP U/L
        {
            "BiomarkerID": 3,
            "RangeType": "longevity",
            "Sex": "M",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 0.80,
            "MaxVal": 1.20,
        },  # Creatinine mg/dL Male
        {
            "BiomarkerID": 3,
            "RangeType": "longevity",
            "Sex": "F",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 0.65,
            "MaxVal": 1.00,
        },  # Creatinine mg/dL Female
        {
            "BiomarkerID": 4,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 80,
            "MaxVal": 95,
        },  # Glucose mg/dL
        {
            "BiomarkerID": 5,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 0.0,
            "MaxVal": 1.0,
        },  # CRP mg/L
        {
            "BiomarkerID": 6,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 4.5,
            "MaxVal": 8.5,
        },  # WBC 10³ cells/µL
        {
            "BiomarkerID": 7,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 25,
            "MaxVal": 40,
        },  # Lymphocyte %
        {
            "BiomarkerID": 8,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 85,
            "MaxVal": 95,
        },  # MCV fL
        {
            "BiomarkerID": 9,
            "RangeType": "longevity",
            "Sex": "All",
            "AgeMin": 20,
            "AgeMax": 30,
            "MinVal": 12.0,
            "MaxVal": 14.0,
        },  # RDW %
    ]

    df = pd.DataFrame(ranges_data)

    # Save to CSV with header comment (no CSV header row)
    output_path = Path("data/clean/reference_ranges.csv")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Reference ranges for 9 Phenotypic Age biomarkers\n")
        f.write(
            "# Units match Measurement table: Glucose mg/dL, CRP mg/L, WBC 10³ cells/µL, etc.\n"
        )
        f.write(
            "# Clinical ranges from Quest Diagnostics 2024; Longevity from NHANES healthy 20-30 subset\n"
        )

    # Append CSV data WITHOUT header row
    df.to_csv(output_path, mode="a", index=False, header=False)

    print(f"✓ Generated reference ranges: {output_path}")
    print(f"✓ Total ranges: {len(df)}")
    print(f"✓ Clinical: {len(df[df.RangeType == 'clinical'])}")
    print(f"✓ Longevity: {len(df[df.RangeType == 'longevity'])}")

    return df


if __name__ == "__main__":
    create_reference_ranges()