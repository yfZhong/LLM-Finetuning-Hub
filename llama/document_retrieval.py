import argparse

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma



directory = './data/docs'

# 加载文档
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# 拆分文档
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

documents = load_docs(directory)
# docs = split_docs(documents)
docs=documents

def main(args):
    documents = load_docs(directory)
    # docs = split_docs(documents)
    docs=documents
    print(len(docs))

    # 嵌入文本
    # embeddings = SentenceTransformerEmbeddings(model_name="chinese-alpaca-2-7b-16k")

    embeddings = HuggingFaceEmbeddings(
      model_name=args.embed_model_id,
      # model_kwargs={
      #   "device": args.device,
      # },
      encode_kwargs={

      },
      # cache_folder=args.cache_dir
    )

    db = Chroma.from_documents(documents=docs, embedding=embeddings, collection_metadata={"hnsw:space": "cosine"})

    query = "What's the sympton of the patient？"
    # matching_docs = db.similarity_search(query)
    matching_docs_with_scores = db.similarity_search_with_relevance_scores(query, args.topk)
    [print(matching_docs_with_scores[i]) for i in range(len(matching_docs_with_scores))]

    print("\n")

    query = "What's the sympton？"
    # matching_docs = db.similarity_search(query)
    matching_docs_with_scores = db.similarity_search_with_relevance_scores(query, args.topk)
    [print(matching_docs_with_scores[i]) for i in range(len(matching_docs_with_scores))]
    print("\n")

    query = "The patient have a fever."
    # matching_docs = db.similarity_search(query)
    matching_docs_with_scores = db.similarity_search_with_relevance_scores(query, args.topk)
    [print(matching_docs_with_scores[i]) for i in range(len(matching_docs_with_scores))]

    a=1


def main2(args):
    documents = load_docs("data/test_docs")
    # docs = split_docs(documents)
    docs=documents
    print(len(docs))

    # 嵌入文本
    # embeddings = SentenceTransformerEmbeddings(model_name="chinese-alpaca-2-7b-16k")

    embeddings = HuggingFaceEmbeddings(
      model_name=args.embed_model_id,
      # model_kwargs={
      #   "device": args.device,
      # },
      encode_kwargs={

      },
      # cache_folder=args.cache_dir
    )

    db = Chroma.from_documents(documents=docs, embedding=embeddings, collection_metadata={"hnsw:space": "cosine"})

    query = "What's the sympton of the patient？"
    # matching_docs = db.similarity_search(query)
    matching_docs_with_scores = db.similarity_search_with_relevance_scores(query, args.topk)
    [print(matching_docs_with_scores[i]) for i in range(len(matching_docs_with_scores))]

    print("\n")

    answer = "The patient feels pain in her head"
    # matching_docs = db.similarity_search(query)
    matching_docs_with_scores = db.similarity_search_with_relevance_scores(answer, args.topk)
    [print(matching_docs_with_scores[i]) for i in range(len(matching_docs_with_scores))]
    print("\n")

    a=1


def rank_based_docs(embeddings, question_list, answer_list, num_query):
    db = Chroma.from_documents(documents=docs, embedding=embeddings, collection_metadata={"hnsw:space": "cosine"})
    scores = []
    for a in answer_list:
        # matching_docs = db.similarity_search(query)
        matching_docs_with_scores = db.similarity_search_with_relevance_scores(a, 1)
        # scores.append([matching_docs_with_scores[0][1], matching_docs_with_scores[0][0].page_content])
        scores.append(matching_docs_with_scores[0][1])

    # Combine docs & scores
    qa_score_pairs = list(zip(question_list, answer_list, scores))

    # Sort by decreasing score
    qa_score_pairs = sorted(qa_score_pairs, key=lambda x: x[2], reverse=True)

    return qa_score_pairs[:num_query]


def rank_docs(embeddings, query, docs):
    db = Chroma.from_documents(documents=docs, embedding=embeddings, collection_metadata={"hnsw:space": "cosine"})
    matching_docs_with_scores = db.similarity_search_with_relevance_scores(query, len(docs))
    return matching_docs_with_scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_id", type=str,
                        default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--output_dir",
                        default="./data/docs/")
    parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()
    # main(args)
    main2(args)