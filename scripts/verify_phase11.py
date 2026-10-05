import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.intelligence.graph_engine import GraphEngine


DATA_PATH = Path("data/processed/engineered.csv")
REPORT_DIR = Path("reports/phase11")


def main():
    print("Phase 11 Graph Intelligence Verification")
    print("=" * 45)

    if not DATA_PATH.exists():
        print("FAIL: engineered dataset not found.")
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    engine = GraphEngine()
    graph = engine.build_graph(df)

    signals = engine.get_graph_signals(
        graph,
        min_relationship_frequency=2,
        min_cluster_size=5,
        min_in_degree=3,
        min_out_degree=3,
    )

    stats = signals["network_statistics"]

    checks = {
        "graph_has_nodes": stats["node_count"] > 0,
        "graph_has_edges": stats["edge_count"] > 0,
        "clusters_detected": stats["cluster_count"] > 0,
        "signals_generated": isinstance(signals, dict),
    }

    report_path = REPORT_DIR / "phase11_graph_intelligence_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("Athena Phase 11 — Graph Intelligence Report\n")
        report.write("=" * 50 + "\n\n")
        report.write(f"Source dataset: {DATA_PATH}\n")
        report.write(f"Nodes: {stats['node_count']}\n")
        report.write(f"Edges: {stats['edge_count']}\n")
        report.write(f"Clusters: {stats['cluster_count']}\n")
        report.write(
            f"Suspicious clusters: "
            f"{stats['suspicious_cluster_count']}\n"
        )
        report.write(
            f"Potential mule accounts: "
            f"{stats['potential_mule_account_count']}\n"
        )
        report.write(
            f"High-frequency relationships: "
            f"{len(signals['high_frequency_relationships'])}\n"
        )
        report.write("\nVerification checks:\n")

        for name, passed in checks.items():
            report.write(f"{name}: {'PASS' if passed else 'FAIL'}\n")

        overall = all(checks.values())
        report.write(f"\nOVERALL: {'PASS' if overall else 'FAIL'}\n")

    if all(checks.values()):
        print("PASS: Phase 11 Graph Intelligence verified.")
        print(f"Report: {report_path}")
        print(stats)
        return 0

    print("FAIL: Phase 11 verification failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())