from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_corpus


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_corpus,
            inputs="raw_corpus",
            outputs="corpus_clean",
            name="clean_corpus_node",
        )
    ])
