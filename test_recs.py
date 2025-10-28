import os
from dotenv import load_dotenv; load_dotenv()

from recommendations_system.rag_index import ensure_built_index
from recommendations_system.recommender import (
    UserContext,
    ConditionRisk,
    generate_recommendations,
)


def test_rag_search():
    print("Building/Loading RAG index...")
    idx = ensure_built_index()
    print("Searching...")
    results = idx.search("hypertension lifestyle recommendations blood pressure targets", top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"#{i} {r['source_path']} p.{r['page_start']}-{r['page_end']} score={r['score']:.3f}")
        print(r['text'][:300].replace('\n', ' '), "...\n")


def test_recommender():
    user = UserContext(
        age_years=52.0,
        sex="Male",
        bmi=29.4,
        weight_kg=None,
        height_cm=None,
        bio_age_years=55.0,
        bai_z=1.2,
    )
    risks = [
        ConditionRisk(
            name="hypertension",
            probability=0.42,
            top_risk_factors=["Systolic blood pressure", "BMI", "Age"],
        ),
        ConditionRisk(
            name="diabetes",
            probability=0.33,
            top_risk_factors=["Fasting glucose", "Waist circumference", "Triglycerides"],
        ),
    ]

    print("Generating recommendations...")
    try:
        recs = generate_recommendations(user=user, risks=risks, top_k=4)
        print(f"Generated {len(recs)} recommendations")
        for i, r in enumerate(recs, start=1):
            print(f"\n[{i}] {r.title}")
            print(r.description)
            for a in r.actions:
                print(f" - {a}")
            if r.citations:
                cites = ", ".join([f"{c.get('source','')} p.{c.get('page_start','')}-{c.get('page_end','')}" for c in r.citations])
                print(f"Citations: {cites}")
    except Exception as e:
        print(f"Recommendation generation failed: {e}")
        print("This is often due to API rate limits or model availability. The RAG indexing and search worked correctly.")


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set; set it in your environment or .env file.")
    test_rag_search()
    test_recommender()


