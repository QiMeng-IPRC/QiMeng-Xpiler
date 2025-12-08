import re

from falcon.src.bm25 import BM25


def retrieve_documentation(query: str, target="mlu"):
    if target == "mlu":
        doc_path = "./falcon/src/bang_api.json"
    else:
        raise RuntimeError("Unsupported target")

    bm25 = BM25(custom_words=[query], doc_path=doc_path)
    ranked_results = bm25.cal_similarity_rank(query)
    strict_results = []

    for doc, score in ranked_results:
        if re.search(r"\b" + re.escape(query) + r"($|[^a-zA-Z0-9_])", doc):
            strict_results.append((doc, score))
    return ranked_results[0][0]


if __name__ == "__main__":
    query = "bang_add"
    print(retrieve_documentation(query))
    print("=======")
    query = "bang_matmul"
    print(retrieve_documentation(query))
