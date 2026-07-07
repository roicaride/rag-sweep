from kedro.pipeline import node, pipeline

from .nodes import chunk_documents, build_index, resolve_params


def create_pipeline(experiment: str = "exp0", carry=None, **kwargs):
    carry = carry or []
    nodes = []

    if "embedding" in carry:
        params_ref = f"{experiment}.params"
        resolve_inputs = [f"params:{experiment}", "best_embedding"]
        if "chunking" in carry:
            resolve_inputs.append("best_chunking")
        nodes.append(node(
            func=resolve_params,
            inputs=resolve_inputs,
            outputs=params_ref,
            name=f"{experiment}_resolve_params",
        ))
    else:
        params_ref = f"params:{experiment}"

    nodes += [
        node(
            func=chunk_documents,
            inputs=["corpus_copyright", params_ref],
            outputs=f"{experiment}.chunks",
            name=f"{experiment}_chunk_documents",
        ),
        node(
            func=build_index,
            inputs=[f"{experiment}.chunks", params_ref, "params:qdrant_url"],
            outputs=f"{experiment}.vectorstore",
            name=f"{experiment}_build_index",
        ),
    ]
    return pipeline(nodes)
