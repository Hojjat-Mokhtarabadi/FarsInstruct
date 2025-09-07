#!/usr/bin/env python3
"""
Demonstration script for the new modular evaluation metrics system.

This script shows how to use BLEU and ROUGE metrics with support for
Arabic, English, and Persian languages.
"""

from metrics import get_metric, Language, MetricsRegistry
from bleu_metric import BLEUMetric
from rouge_metric import ROUGEMetric


def demo_bleu_metric():
    """Demonstrate BLEU metric with different languages."""
    print("=== BLEU Metric Demonstration ===\n")

    # Sample texts in different languages
    english_refs = ["The quick brown fox jumps over the lazy dog."]
    english_preds = ["The fast brown fox jumps over the lazy dog."]

    persian_refs = ["روباه قهوه‌ای سریع از روی سگ تنبل می‌پرد."]
    persian_preds = ["روباه تیره سریع از روی سگ خواب می‌پرد."]

    arabic_refs = ["الثعلب البني السريع يقفز فوق الكلب الكسول."]
    arabic_preds = ["الثعلب السريع يقفز فوق الكلب الكسول."]

    # Test with different languages
    languages = [
        (Language.ENGLISH, "English", english_refs, english_preds),
        (Language.PERSIAN, "Persian", persian_refs, persian_preds),
        (Language.ARABIC, "Arabic", arabic_refs, arabic_preds),
        (Language.AUTO, "Auto-detect", english_refs, english_preds)
    ]

    for lang, lang_name, refs, preds in languages:
        print(f"--- {lang_name} ({lang.value}) ---")

        bleu = BLEUMetric(language=lang)
        result = bleu.compute(preds, refs)

        print(f"BLEU Score: {result.scores['bleu'].fmeasure:.4f}")
        print(f"Language detected: {result.language.value}")
        print()


def demo_rouge_metric():
    """Demonstrate ROUGE metric with different languages."""
    print("=== ROUGE Metric Demonstration ===\n")

    # Sample texts in different languages
    english_refs = ["The quick brown fox jumps over the lazy dog."]
    english_preds = ["The fast brown fox leaps over the sleeping dog."]

    persian_refs = ["روباه قهوه‌ای سریع از روی سگ تنبل می‌پرد."]
    persian_preds = ["روباه تیره سریع از روی سگ خواب می‌پرد."]

    # Test ROUGE with Persian
    print("--- Persian ROUGE ---")
    rouge = ROUGEMetric(language=Language.PERSIAN)
    result = rouge.compute(persian_preds, persian_refs)

    for metric_name, score in result.scores.items():
        print(f"{metric_name}:")
        print(".2f")
        print(".2f")
        print(".2f")
    print(f"Language: {result.language.value}")
    print()


def demo_factory_pattern():
    """Demonstrate the factory pattern for easy metric creation."""
    print("=== Factory Pattern Demonstration ===\n")

    # Sample data
    predictions = ["The cat sits on the mat."]
    references = ["The feline rests on the rug."]

    # Create metrics using factory function
    metrics_to_test = ['bleu', 'rouge']

    for metric_name in metrics_to_test:
        print(f"--- {metric_name.upper()} via Factory ---")

        # Create metric with auto language detection
        metric = get_metric(metric_name, language=Language.AUTO)
        result = metric.compute(predictions, references)

        print(f"Metric: {metric_name}")
        print(f"Language: {result.language.value}")
        print(f"Results: {result.to_dict()}")
        print()


def demo_metric_registry():
    """Demonstrate the metrics registry."""
    print("=== Metrics Registry Demonstration ===\n")

    print("Available metrics:")
    for metric_name in MetricsRegistry.list_metrics():
        print(f"  - {metric_name}")

    print("\nRegistry contains:")
    for name, metric_class in MetricsRegistry._metrics.items():
        print(f"  {name}: {metric_class.__name__}")

    print()


def main():
    """Run all demonstrations."""
    print("🔥 FarsInstruct Modular Metrics System Demo 🔥\n")
    print("This demo showcases the new evaluation metrics system with")
    print("multi-language support for Arabic, English, and Persian.\n")

    try:
        demo_metric_registry()
        demo_bleu_metric()
        demo_rouge_metric()
        demo_factory_pattern()

        print("✅ All demonstrations completed successfully!")
        print("\n📚 Key Features:")
        print("  • Modular design for easy metric extension")
        print("  • Multi-language support (Arabic, English, Persian)")
        print("  • Factory pattern for easy metric creation")
        print("  • Backward compatibility with existing code")
        print("  • Auto language detection")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

d