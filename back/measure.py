import ir_datasets
import bm25s
import Stemmer

# Snowball Stemmer initialize kiya (English language)
stemmer = Stemmer.Stemmer("english")

def run_lexical_baseline():
    # 1. Dataset download & load (BEIR benchmark ka SciFact test split)
    print("Loading beir/scifact/test...")
    dataset = ir_datasets.load("beir/scifact/test")

    doc_ids = []
    corpus_text = []
    doc_map = {}  # doc_id -> raw text ka key-value lookup
    
    # 2. Corpus iteration: Har paper ka Title aur Abstract text joda
    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        text = f"{doc.title} {doc.text}".strip()
        corpus_text.append(text)
        doc_map[doc.doc_id] = text

    print(f"Total documents loaded: {len(corpus_text)}")

    # 3. Tokenization & Stemming:
    # Text ko words me toda, English stop-words hataye, aur har word ko uske root form me reduce kiya
    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en", stemmer=stemmer)

    # 4. Inverted Index create karna (k1=1.5 term saturation, b=0.75 length penalty)
    # corpus=doc_ids pass kiya taaki search results direct document ID return karein
    retriever = bm25s.BM25(corpus=doc_ids, k1=1.5, b=0.75)
    retriever.index(corpus_tokens)
    print("Indexing complete.")

    # 5. Queries fetch karna (Sample ke liye pehli 3 queries)
    sample_queries = list(dataset.queries_iter())[:3]

    for q in sample_queries:
        query_text = q.text
        # Query ko bhi usi stemmer se process kiya (taaki vocabulary mismatch na ho)
        query_tokens = bm25s.tokenize(query_text, stemmer=stemmer)

        # 6. Retrieval: Top 5 sabse relevant document IDs aur unke BM25 scores nikale
        results, scores = retriever.retrieve(query_tokens, k=5)

        print(f"\n--- Query ID: {q.query_id} ---")
        print(f"Query: {query_text}")
        print("Top Matches:")
        for rank, (matched_doc_id, score) in enumerate(zip(results[0], scores[0]), start=1):
            preview = doc_map[matched_doc_id][:90].replace("\n", " ")
            print(f"  [{rank}] Doc ID: {matched_doc_id} | BM25 Score: {score:.4f} | Snippet: {preview}...")