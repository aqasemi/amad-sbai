## Recommendations System

This module provides a simple RAG (retrieval-augmented generation) pipeline over local guideline PDFs (Saudi MoH and WHO) and uses Gemini 1.5 Flash to produce tailored, cited recommendations.

### Layout

- `guidelines/`: place PDF guideline documents here
- `rag_index.py`: PDF loader, chunker, and embedding-based vector store using `text-embedding-004`
- `recommender.py`: builds a focused query, retrieves top-k chunks, prompts Gemini, and returns structured JSON recommendations

### Setup

1. Install dependencies:
```bash
pip install -r /home/ubuntu/datathon/recommendations_system/requirements.txt
# or with uv:
# uv pip install -r /home/ubuntu/datathon/recommendations_system/requirements.txt
```

2. Set your Google API key (Gemini):
```bash
export GOOGLE_API_KEY=YOUR_KEY_HERE
```
Or create a `.env` file in the project root with:
```
GOOGLE_API_KEY=YOUR_KEY_HERE
```

3. Ensure guideline PDFs are present under:
```
/home/ubuntu/datathon/recommendations_system/guidelines
```

The first run will build an index at:
```
/home/ubuntu/datathon/recommendations_system/index/guidelines_index.pkl
```
It is automatically rebuilt if PDFs change.

### Testing

Run the test script to verify RAG indexing, search, and recommendation generation:
```bash
python3 test_recs.py
```

This will:
- Build/load the RAG index from guidelines PDFs
- Run a sample search query
- Generate personalized recommendations based on dummy risk profiles

### Usage in Streamlit app

The app imports:

```python
from recommendations_system.recommender import UserContext, ConditionRisk, generate_recommendations
```

It constructs `UserContext` and a list of `ConditionRisk` from computed bioage and risk model outputs, then:
1. **Checks for existing recommendations** by patient SEQN
2. If found, loads the most recent file (sorted by modification time)
3. If not found, shows a 3-second spinner while generating new recommendations
4. Saves new recommendations to `/home/ubuntu/datathon/recommendations_output/recommendations_SEQN_{patient_id}_{timestamp}.json`
5. Renders recommendation cards with citations
6. Provides a "🔄 Regenerate Recommendations" button if using cached recommendations

Each saved file includes:
- Patient ID (SEQN)
- Timestamp
- Patient demographics (age, sex, BMI, bio age, BAI)
- Risk probabilities and top factors
- Full recommendations with actions and citations

### Notes

- Embedding model: `text-embedding-004`
- Generation model: `gemini-1.5-flash-002`
- Long PDF chunks are truncated at ~8k chars for embedding safety.


