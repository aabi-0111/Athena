"""
Athena
--------------------
Graph-Based Fraud Intelligence Engine

Phase 11 responsibilities:
1. Represent transaction entities.
2. Represent transaction relationships.
3. Link transactions between accounts.
4. Preserve relationship frequency.
5. Identify high-frequency relationships.
6. Do not perform cluster detection or fraud scoring at this stage.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from app.core.constants import (
    COL_AMOUNT,
    COL_NAME_DEST,
    COL_NAME_ORIG,
    COL_STEP,
    COL_TYPE,
)


class GraphEngine:
    """Build and expose the PaySim transaction relationship graph."""

    REQUIRED_COLUMNS = (
        COL_NAME_ORIG,
        COL_NAME_DEST,
        COL_STEP,
        COL_TYPE,
        COL_AMOUNT,
    )

    def build_graph(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """
        Build a directed transaction graph from a PaySim DataFrame.

        Nodes represent sender/receiver accounts.

        Each transaction is represented by a directed edge:
            sender -> receiver

        Repeated sender/receiver relationships are preserved as
        separate transaction edges and their relationship frequency
        is explicitly stored on every edge.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "GraphEngine.build_graph() expects a pandas DataFrame."
            )

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required graph columns: {missing}"
            )

        graph = nx.MultiDiGraph()

        # ---------------------------------------------------------
        # Calculate relationship frequency
        # ---------------------------------------------------------
        relationship_frequency = (
            df.groupby(
                [COL_NAME_ORIG, COL_NAME_DEST],
                dropna=False,
            )
            .size()
            .to_dict()
        )

        # ---------------------------------------------------------
        # Build transaction graph
        # ---------------------------------------------------------
        for transaction_id, row in enumerate(
            df.itertuples(index=False),
            start=1,
        ):
            sender = getattr(row, COL_NAME_ORIG)
            receiver = getattr(row, COL_NAME_DEST)

            frequency = relationship_frequency[
                (sender, receiver)
            ]

            # -----------------------------------------------------
            # Entity representation
            # -----------------------------------------------------
            graph.add_node(
                sender,
                entity_type="account",
            )

            graph.add_node(
                receiver,
                entity_type="account",
            )

            # -----------------------------------------------------
            # Transaction relationship representation
            # -----------------------------------------------------
            graph.add_edge(
                sender,
                receiver,
                transaction_id=transaction_id,
                step=getattr(row, COL_STEP),
                transaction_type=getattr(row, COL_TYPE),
                amount=getattr(row, COL_AMOUNT),
                relationship_frequency=frequency,
            )

        return graph

    def get_high_frequency_relationships(
        self,
        graph: nx.MultiDiGraph,
        min_frequency: int = 2,
    ) -> list[dict]:
        """
        Return sender-receiver relationships whose transaction
        frequency meets or exceeds the supplied threshold.

        This method does not assign fraud labels or calculate
        fraud scores.
        """

        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                "Expected a networkx.MultiDiGraph."
            )

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be at least 1."
            )

        relationships = {}

        for sender, receiver, data in graph.edges(data=True):
            key = (sender, receiver)

            frequency = data["relationship_frequency"]

            if key not in relationships:
                relationships[key] = frequency

        return [
            {
                "sender": sender,
                "receiver": receiver,
                "relationship_frequency": frequency,
            }
            for (sender, receiver), frequency in relationships.items()
            if frequency >= min_frequency
        ]
        
    def get_transaction_clusters(
        self,
        graph: nx.MultiDiGraph,
    ) -> list[dict]:
        """
        Identify connected transaction clusters efficiently.

        Accounts are grouped into the same cluster when they are connected
        through transaction relationships, regardless of transaction
        direction.

        No fraud score or fraud classification is assigned here.
        """

        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                "Expected a networkx.MultiDiGraph."
            )

        clusters = []

        for cluster_id, nodes in enumerate(
            nx.weakly_connected_components(graph),
            start=1,
        ):
            nodes = list(nodes)

            cluster_graph = graph.subgraph(nodes)

            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "account_count": len(nodes),
                    "accounts": nodes,
                    "transaction_count": cluster_graph.number_of_edges(),
                }
            )

        return clusters

    def get_potential_mule_accounts(
        self,
        graph: nx.MultiDiGraph,
        min_in_degree: int = 3,
        min_out_degree: int = 3,
        min_total_degree: int = 6,
    ) -> list[dict]:
        """
        Identify accounts showing a potential mule-account network pattern.

        This is a graph-derived intelligence signal only.
        It does not classify an account as fraudulent.
        """

        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                "Expected a networkx.MultiDiGraph."
            )

        if min_in_degree < 1:
            raise ValueError(
                "min_in_degree must be at least 1."
            )

        if min_out_degree < 1:
            raise ValueError(
                "min_out_degree must be at least 1."
            )

        if min_total_degree < 1:
            raise ValueError(
                "min_total_degree must be at least 1."
            )

        mule_accounts = []

        for account in graph.nodes:
            in_degree = graph.in_degree(account)
            out_degree = graph.out_degree(account)
            total_degree = in_degree + out_degree

            if (
                in_degree >= min_in_degree
                and out_degree >= min_out_degree
                and total_degree >= min_total_degree
            ):
                mule_accounts.append(
                    {
                        "account": account,
                        "incoming_transactions": in_degree,
                        "outgoing_transactions": out_degree,
                        "total_transactions": total_degree,
                        "mule_network_signal": True,
                    }
                )

        return mule_accounts

    def get_graph_signals(
        self,
        graph: nx.MultiDiGraph,
        min_relationship_frequency: int = 2,
        min_cluster_size: int = 5,
        min_in_degree: int = 3,
        min_out_degree: int = 3,
    ) -> dict:
        """
        Generate structured graph-derived intelligence signals.

        Signals include:
        - repeated/high-frequency relationships
        - connected transaction clusters
        - potential mule-account patterns
        - network-level connectivity indicators

        These signals do not produce a fraud score or final decision.
        """

        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                "Expected a networkx.MultiDiGraph."
            )

        high_frequency_relationships = (
            self.get_high_frequency_relationships(
                graph,
                min_frequency=min_relationship_frequency,
            )
        )

        clusters = self.get_transaction_clusters(graph)

        suspicious_clusters = [
            cluster
            for cluster in clusters
            if cluster["account_count"] >= min_cluster_size
        ]

        potential_mule_accounts = (
            self.get_potential_mule_accounts(
                graph,
                min_in_degree=min_in_degree,
                min_out_degree=min_out_degree,
                min_total_degree=(
                    min_in_degree + min_out_degree
                ),
            )
        )

        return {
            "high_frequency_relationships": (
                high_frequency_relationships
            ),
            "suspicious_clusters": suspicious_clusters,
            "potential_mule_accounts": potential_mule_accounts,
            "network_statistics": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "cluster_count": len(clusters),
                "suspicious_cluster_count": len(
                    suspicious_clusters
                ),
                "potential_mule_account_count": len(
                    potential_mule_accounts
                ),
            },
        }