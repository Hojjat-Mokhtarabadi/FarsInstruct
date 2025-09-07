import unittest

from FarsInstruct.evaluation.metrics import build_metrics


class TestMetricsMultilingual(unittest.TestCase):
    def test_rouge_bleu_english_identity(self):
        preds = ["The quick brown fox jumps over the lazy dog."]
        refs = ["The quick brown fox jumps over the lazy dog."]
        metrics = build_metrics(["rouge", "bleu"], lang="en")
        rouge = metrics["rouge"].compute(preds, refs)
        bleu = metrics["bleu"].compute(preds, refs)

        # ROUGE-L F1 close to 1
        self.assertIn("rougeL", rouge)
        self.assertAlmostEqual(rouge["rougeL"].mid.fmeasure, 1.0, places=6)
        # BLEU 100 for identical strings
        self.assertGreaterEqual(bleu["bleu"], 99.99)

    def test_rouge_bleu_arabic_identity(self):
        preds = ["سريعٌ الثعلبُ البنيُّ يقفز فوق الكلب الكسول."]
        refs = ["سريعٌ الثعلبُ البنيُّ يقفز فوق الكلب الكسول."]
        metrics = build_metrics(["rouge", "bleu"], lang="ar")
        rouge = metrics["rouge"].compute(preds, refs)
        bleu = metrics["bleu"].compute(preds, refs)

        self.assertIn("rougeL", rouge)
        self.assertAlmostEqual(rouge["rougeL"].mid.fmeasure, 1.0, places=6)
        self.assertGreaterEqual(bleu["bleu"], 99.99)

    def test_rouge_bleu_mismatch_lower_scores(self):
        preds = ["hello world"]
        refs = ["goodbye universe"]
        metrics = build_metrics(["rouge", "bleu"], lang="en")
        rouge = metrics["rouge"].compute(preds, refs)
        bleu = metrics["bleu"].compute(preds, refs)

        # Expect near-zero
        self.assertLess(rouge["rougeL"].mid.fmeasure, 0.2)
        self.assertLess(bleu["bleu"], 20.0)

    def test_rouge_bleu_persian_identity_and_mismatch(self):
        # Identity
        p_pred = ["روباه قهوه‌ای تند و تیز از روی سگ تنبل می‌پرد."]
        p_ref = ["روباه قهوه‌ای تند و تیز از روی سگ تنبل می‌پرد."]
        metrics_fa = build_metrics(["rouge", "bleu"], lang="fa")
        rouge_fa = metrics_fa["rouge"].compute(p_pred, p_ref)
        bleu_fa = metrics_fa["bleu"].compute(p_pred, p_ref)
        self.assertAlmostEqual(rouge_fa["rougeL"].mid.fmeasure, 1.0, places=6)
        self.assertGreaterEqual(bleu_fa["bleu"], 99.99)

        # Mismatch
        p_pred2 = ["سلام دنیا"]
        p_ref2 = ["شب خوش"]
        rouge_fa2 = metrics_fa["rouge"].compute(p_pred2, p_ref2)
        bleu_fa2 = metrics_fa["bleu"].compute(p_pred2, p_ref2)
        self.assertLess(rouge_fa2["rougeL"].mid.fmeasure, 0.2)
        self.assertLess(bleu_fa2["bleu"], 20.0)


if __name__ == "__main__":
    unittest.main()


