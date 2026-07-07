from kedro.pipeline import Pipeline, node, pipeline
from .nodes import sample_documents, build_knowledge_graph, generate_qa_pairs


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=sample_documents,
            inputs=["corpus_copyright", "params:gold_dataset"],
            outputs="gold_candidates",
            name="sample_documents_node",
        ),
        node(
            func=build_knowledge_graph,
            inputs=["gold_candidates", "params:gold_dataset"],
            outputs="knowledge_graph",
            name="build_knowledge_graph_node",
        ),
        node(
            func=generate_qa_pairs,
            inputs=["knowledge_graph", "params:gold_dataset"],
            outputs="gold_dataset",
            name="generate_qa_pairs_node",
        ),
    ])
