import json
import math
import os
import random
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import SingleTurnSample
from ragas.run_config import RunConfig
from ragas.testset import Testset, TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.base import MultiHopScenario
from ragas.testset.synthesizers.multi_hop.prompts import QueryConditions
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    CustomNodeFilter,
    OverlapScoreBuilder,
    Parallel,
    apply_transforms,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.utils import num_tokens_from_string
import ragas.testset.synthesizers.generate as _ragas_gen


def _patch_ragas_nan_filter():
    _OriginalSample = _ragas_gen.TestsetSample

    class _SafeFactory:
        def __call__(self, eval_sample=None, **kwargs):
            if isinstance(eval_sample, float) and math.isnan(eval_sample):
                return None
            return _OriginalSample(eval_sample=eval_sample, **kwargs)

    _ragas_gen.TestsetSample = _SafeFactory()

    _orig_init = Testset.__init__

    def _patched_init(self, samples=None, **kwargs):
        if samples:
            samples = [s for s in samples if s is not None]
        _orig_init(self, samples=samples, **kwargs)

    Testset.__init__ = _patched_init


_patch_ragas_nan_filter()

@dataclass
class MultiHopSpecificQuerySynthesizerFixed(MultiHopSpecificQuerySynthesizer):
    """Corrixe dous problemas do synthesizer base de RAGAS:
    - prepare_combinations descarta nós cuxo nome de entidade difire do tema de overlap
      (fuzzy match na detección vs exact match no filtrado), deixando só
      1 contexto nas mostras multi-hop. Forzamos a manter sempre os dous nós orixinais.
    - _generate_sample non enche persona_name/query_style/query_length, polo que
      eses campos saen como NaN. Enchémolos aquí.
    """

    def prepare_combinations(self, nodes, combinations, personas, persona_item_mapping, property_name):
        samples = super().prepare_combinations(nodes, combinations, personas, persona_item_mapping, property_name)
        for s in samples:
            if len(s["nodes"]) < 2:
                s["nodes"] = list(nodes)
        return samples

    async def _generate_sample(self, scenario, callbacks):
        if not isinstance(scenario, MultiHopScenario):
            raise TypeError(f"Expected MultiHopScenario, got {type(scenario)}")
        reference_contexts = self.make_contexts(scenario)
        response = await self.generate_query_reference_prompt.generate(
            data=QueryConditions(
                persona=scenario.persona,
                themes=scenario.combinations,
                context=reference_contexts,
                query_length=scenario.length.value,
                query_style=scenario.style.value,
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=reference_contexts,
            persona_name=getattr(scenario.persona, "name", None),
            query_style=scenario.style.value,
            query_length=scenario.length.value,
        )


_SYNTHESIZER_MAP = {
    "single_hop_specific_query_synthesizer": SingleHopSpecificQuerySynthesizer,
    "multi_hop_specific_query_synthesizer":  MultiHopSpecificQuerySynthesizerFixed,
}

_PERSONAS = [
    Persona(
        name="IP Lawyer",
        role_description="Specialises in intellectual property and copyright law, focused on legal precedents and statutory interpretation.",
    ),
    Persona(
        name="Policy Analyst",
        role_description="Analyses EU digital policy and legislation, interested in the societal impact of copyright rules.",
    ),
    Persona(
        name="Software Developer",
        role_description="Works with open-source software and needs to understand copyright implications for code and digital works.",
    ),
]


def _make_llm(llm_params: dict) -> ChatOpenAI:
    return ChatOpenAI(
        model=llm_params["model"],
        temperature=llm_params["temperature"],
        max_tokens=llm_params["max_tokens"],
        base_url=llm_params["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )


def _build_transforms(llm, langchain_docs: list):
    def is_long_doc(node):
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties["page_content"]) > 500
        )

    def has_headlines(node):
        return node.get_property("headlines") is not None

    def is_chunk(node):
        return node.type == NodeType.CHUNK

    frac_long = sum(
        1 for d in langchain_docs if num_tokens_from_string(d.page_content) > 500
    ) / len(langchain_docs)

    if frac_long >= 0.25:
        return [
            HeadlinesExtractor(llm=llm, filter_nodes=is_long_doc),
            HeadlineSplitter(min_tokens=500, filter_nodes=has_headlines),
            CustomNodeFilter(llm=llm, filter_nodes=is_chunk),
            Parallel(
                ThemesExtractor(llm=llm, filter_nodes=is_chunk),
                NERExtractor(llm=llm, filter_nodes=is_chunk),
            ),
            OverlapScoreBuilder(threshold=0.01, filter_nodes=is_chunk),
        ]
    else:
        return [
            CustomNodeFilter(llm=llm),
            Parallel(
                ThemesExtractor(llm=llm, filter_nodes=lambda n: n.type == NodeType.DOCUMENT),
                NERExtractor(llm=llm),
            ),
            OverlapScoreBuilder(threshold=0.01),
        ]


def sample_documents(corpus_copyright: list[str], params: dict) -> list[dict]:
    seed   = params["seed"]
    strata = params["strata"]

    docs = [json.loads(l) for l in corpus_copyright if l.strip()]

    random.seed(seed)
    sampled = []
    for stratum in strata:
        lo = stratum["min_words"]
        hi = stratum.get("max_words") or float("inf")
        n  = stratum["n"]
        pool = [d for d in docs if lo <= len(d["text"].split()) < hi]
        k = min(n, len(pool))
        sampled.extend(random.sample(pool, k))
        label = f"{lo}-{int(hi) if hi != float('inf') else 'inf'}"
        print(f"  [{label} words] pool={len(pool)}, sampled={k}")

    print(f"Total mostreado: {len(sampled)}")
    return sampled


def build_knowledge_graph(documents: list[dict], params: dict) -> KnowledgeGraph:
    lc_llm = _make_llm(params["llm"])
    lc_emb = GoogleGenerativeAIEmbeddings(
        model=params["embedding_model"],
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    generator = TestsetGenerator.from_langchain(llm=lc_llm, embedding_model=lc_emb)

    langchain_docs = [
        Document(
            page_content=doc["text"],
            metadata={k: v for k, v in doc.items() if k != "text"},
        )
        for doc in documents
    ]

    kg = KnowledgeGraph(nodes=[
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata},
        )
        for doc in langchain_docs
    ])

    transforms = _build_transforms(generator.llm, langchain_docs)
    run_config = RunConfig(max_workers=params["max_workers"])

    print(f"Aplicando transforms a {len(langchain_docs)} docs...")
    apply_transforms(kg, transforms, run_config=run_config)
    print("Transforms completados.")
    return kg


def generate_qa_pairs(knowledge_graph: KnowledgeGraph, params: dict) -> list[dict]:
    lc_llm = _make_llm(params["llm"])
    lc_emb = GoogleGenerativeAIEmbeddings(
        model=params["embedding_model"],
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    generator = TestsetGenerator.from_langchain(llm=lc_llm, embedding_model=lc_emb)
    generator.knowledge_graph = knowledge_graph
    generator.persona_list = _PERSONAS

    query_distribution = None
    if "query_distribution" in params:
        query_distribution = [
            (_SYNTHESIZER_MAP[name](llm=lc_llm), weight)
            for name, weight in params["query_distribution"].items()
        ]

    run_config = RunConfig(max_workers=params["max_workers"])

    print(f"Xerando {params['testset_size']} pares QA...")
    testset = generator.generate(
        testset_size=params["testset_size"],
        query_distribution=query_distribution,
        run_config=run_config,
        num_personas=2,
        raise_exceptions=False,
    )

    df = testset.to_pandas()
    print(f"Xerados {len(df)} pares QA")
    print(df["synthesizer_name"].value_counts().to_string())
    return df.to_dict(orient="records")
