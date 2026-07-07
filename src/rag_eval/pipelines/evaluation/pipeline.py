from kedro.pipeline import node, pipeline

from .nodes import run_rag, evaluate_ragas


def create_pipeline(experiment: str = "exp0", carry=None, source=None, **kwargs):
    carry = carry or []
    p = f"{experiment}.params" if "embedding" in carry else f"params:{experiment}"
    vs = source if source else f"{experiment}.vectorstore"  # source: índice compartido (exp3 reutiliza o da config gañadora)
    return pipeline([
        node(
            func=run_rag,
            inputs=[vs, "gold_dataset", p, "params:rag_prompt", "params:qdrant_url"],
            outputs=f"{experiment}.rag_results",
            name=f"{experiment}_run_rag",
        ),
        node(
            func=evaluate_ragas,
            inputs=[f"{experiment}.rag_results", p, "params:mlflow_experiment", "params:ragas_run_config", "params:ragas_embedding"],
            outputs=f"{experiment}.eval_metrics",
            name=f"{experiment}_evaluate_ragas",
        ),
    ])
