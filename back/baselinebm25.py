import ir_datasets
import bm25s
import Stemmer

# Initialize Snowball stemmer once to ensure identical morphological normalization
# across indexing and query-time tokenization.
stemmer = Stemmer.Stemmer("english")

def run_lexical_baseline():
    print("Loading beir/scifact/test...")
    dataset = ir_datasets.load("beir/scifact/test")

    doc_ids = []
    corpus_text = []
    doc_map = {}  # Fast snippet lookup for document text by doc_id
    
    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        # Concatenation of title and abstract for full lexical surface coverage
        text = f"{doc.title} {doc.text}".strip()
        corpus_text.append(text)
        doc_map[doc.doc_id] = text

    print(f"Total documents loaded: {len(corpus_text)}")

    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en", stemmer=stemmer)

    # Passing doc_ids directly to corpus maps retrieval indices to string IDs
    retriever = bm25s.BM25(corpus=doc_ids, k1=1.5, b=0.75)
    retriever.index(corpus_tokens)
    print("Indexing complete.")

    sample_queries = list(dataset.queries_iter())[:3]

    for q in sample_queries:
        query_text = q.text
        query_tokens = bm25s.tokenize(query_text, stemmer=stemmer)

        # results[0] contains doc_ids directly
        results, scores = retriever.retrieve(query_tokens, k=5)

        print(f"\n--- Query ID: {q.query_id} ---")
        print(f"Query: {query_text}")
        print("Top Matches:")
        for rank, (matched_doc_id, score) in enumerate(zip(results[0], scores[0]), start=1):
            preview = doc_map[matched_doc_id][:90].replace("\n", " ")
            print(f"  [{rank}] Doc ID: {matched_doc_id} | BM25 Score: {score:.4f} | Snippet: {preview}...")

        # Format: {query_id: {doc_id: relevance_score}}
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

if __name__ == "__main__":
    run_lexical_baseline()
