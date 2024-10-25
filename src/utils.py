import os

import nest_asyncio
import numpy as np
from dotenv import find_dotenv, load_dotenv
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    get_leaf_nodes,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from trulens_eval import Feedback, OpenAI, TruLlama

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


openai = OpenAI()

qa_relevance = Feedback(
    openai.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

# grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
groundedness = (
    Feedback(openai.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)
    return tru_recorder


def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)
    return tru_recorder


def build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document],
            embed_model=embed_model,
            node_parser=node_parser,
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            embed_model=embed_model,
            llm=llm,
        )

    return sentence_index


def get_sentence_window_query_engine(
    llm,
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
    )
    return sentence_window_engine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            llm=llm,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            embed_model=embed_model,
            llm=llm,
        )
    return automerging_index


def get_automerging_query_engine(
    llm,
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k,
    )
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[rerank],
        llm=llm,
    )
    return auto_merging_engine
