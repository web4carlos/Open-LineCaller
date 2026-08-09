import argparse
from linecaller.validation.io import load_truth_jsonl, load_predictions_jsonl
from linecaller.validation.comparison import compare_events
from linecaller.validation.metrics import compute_metrics
from linecaller.validation.report import export_validation_report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()

    truth = load_truth_jsonl(args.ground_truth)
    pred = load_predictions_jsonl(args.predictions)
    comparisons = compare_events(truth, pred)
    metrics = compute_metrics(comparisons)
    export_validation_report(metrics, comparisons, args.report)

    print(f"automatic_accuracy={metrics.automatic_accuracy:.4f}")
    print(f"coverage={metrics.coverage:.4f}")
    print(f"review_rate={metrics.review_rate:.4f}")
    print(f"miss_rate={metrics.miss_rate:.4f}")
    print(f"false_in={metrics.false_in}")
    print(f"false_out={metrics.false_out}")

if __name__ == "__main__":
    main()
