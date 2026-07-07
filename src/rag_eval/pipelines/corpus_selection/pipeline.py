from kedro.pipeline import Pipeline, node, pipeline
from .nodes import select_copyright


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=select_copyright,
            inputs=["corpus_clean", "params:corpus_selection"],
            outputs="corpus_copyright",
            name="select_copyright_node",
        )
    ])
