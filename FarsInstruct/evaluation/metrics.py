import typing as t


class BaseMetric:
    """Abstract metric interface."""

    def compute(self, predictions: t.List[str], references: t.List[str]) -> t.Dict[str, t.Any]:
        raise NotImplementedError


class RougeMetric(BaseMetric):
    """Wrapper around local ROUGE with language support."""

    def __init__(self, lang: str = "fa", rouge_types: t.Optional[t.List[str]] = None, use_stemmer: bool = False):
        import datasets
        import os
        metric_dir = os.path.join(os.path.dirname(__file__), "rouge")
        self.metric = datasets.load_metric(metric_dir)
        self.lang = lang
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.use_stemmer = use_stemmer

    def compute(self, predictions: t.List[str], references: t.List[str]) -> t.Dict[str, t.Any]:
        # The local ROUGE implementation expects detokenized strings; sentence breaks are handled upstream if needed.
        for pred, ref in zip(predictions, references):
            self.metric.add_batch(predictions=[pred], references=[ref])
        scores = self.metric.compute(rouge_types=self.rouge_types, use_stemmer=self.use_stemmer, lang=self.lang)
        return scores


class BleuMetric(BaseMetric):
    """BLEU via sacrebleu with reasonable defaults per language."""

    def __init__(self, lang: str = "fa", tokenize: t.Optional[str] = None):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.lang = lang
        if tokenize is not None:
            self.tokenize = tokenize
        else:
            # Map language to sacreBLEU tokenization
            # 'intl' is better for languages using non-Latin scripts
            self.tokenize = {
                "en": "13a",
                "fa": "intl",
                "ar": "intl",
            }.get(lang, "intl")

    def compute(self, predictions: t.List[str], references: t.List[str]) -> t.Dict[str, t.Any]:
        # sacrebleu expects list of hypothesis strings and list of reference lists
        refs = [references]
        result = self.sacrebleu.corpus_bleu(predictions, refs, tokenize=self.tokenize)
        return {
            "bleu": result.score,
            "precisions": result.precisions,
            "bp": result.bp,
            "sys_len": result.sys_len,
            "ref_len": result.ref_len,
        }


def build_metrics(metric_names: t.List[str], lang: str) -> t.Dict[str, BaseMetric]:
    registry: t.Dict[str, BaseMetric] = {}
    for name in metric_names:
        key = name.strip().lower()
        if key == "rouge":
            registry[key] = RougeMetric(lang=lang)
        elif key == "bleu":
            registry[key] = BleuMetric(lang=lang)
        else:
            raise ValueError(f"Unknown metric: {name}")
    return registry


