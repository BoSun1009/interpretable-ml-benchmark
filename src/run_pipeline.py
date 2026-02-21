"""
Modular Interpretable ML Benchmarking Pipeline
----------------------------------------------

This script demonstrates:
- Multiple feature selection strategies
- Feature Co-occurrence Network (FCN) integration
- PageRank-based global feature ranking
- Unified XGBoost evaluation

Designed for reproducible benchmarking on structured datasets.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# ==========================
# Global Configuration
# ==========================
TOP_K
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# ==========================
# Dataset Loader
# ==========================
def load_dataset():
    """Load example structured dataset (replaceable)."""
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names
    return X, y, feature_names


# ==========================
# Feature Selection Methods
# ==========================
def select_chi2(X, y, k):
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True)


def select_mutual_info(X, y, k):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True)


def select_rfe(X, y, k):
    estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    return np.where(selector.support_)[0]


# ==========================
# Feature Co-occurrence Network
# ==========================
def build_feature_cooccurrence_network(feature_sets):
    """
    Construct graph where:
    Nodes = feature indices
    Edge weight = frequency of co-selection
    """
    unique_features = list(set([f for subset in feature_sets for f in subset]))
    graph = nx.Graph()
    graph.add_nodes_from(unique_features)

    for subset in feature_sets:
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if graph.has_edge(subset[i], subset[j]):
                    graph[subset[i]][subset[j]]["weight"] += 1
                else:
                    graph.add_edge(subset[i], subset[j], weight=1)

    return graph


# ==========================
# Graph-Based Ranking
# ==========================
def rank_features_via_pagerank(graph, top_k):
    pagerank_scores = nx.pagerank(graph, weight="weight")
    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in ranked[:top_k]]
    return selected, pagerank_scores


# ==========================
# Model Evaluation
# ==========================
def evaluate_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }


# ==========================
# Visualisation
# ==========================
def visualise_network(graph, selected_features):
    subgraph = graph.subgraph(selected_features)
    pos = nx.spring_layout(subgraph, seed=RANDOM_STATE)

    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=800,
        font_size=8,
    )

    plt.title("Feature Co-occurrence Network (Top Features)")
    plt.tight_layout()
    plt.show()


# ==========================
# Main Execution
# ==========================
def main():

    # Load dataset
    X, y, feature_names = load_dataset()
    X_scaled = MinMaxScaler().fit_transform(X)

    # Run feature selection strategies
    feature_selection_results = [
        select_chi2(X_scaled, y, TOP_K),
        select_mutual_info(X_scaled, y, TOP_K),
        select_rfe(X, y, TOP_K),
    ]

    # Build integration graph
    graph = build_feature_cooccurrence_network(feature_selection_results)

    # Rank features globally
    selected_features, pagerank_scores = rank_features_via_pagerank(graph, TOP_K)

    # Evaluate final model
    evaluation_metrics = evaluate_xgboost(X[:, selected_features], y)

    print("\nTop Features (PageRank Ranked):")
    for idx in selected_features:
        print(f"- {feature_names[idx]}")

    print("\nEvaluation Results:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Visualise
    visualise_network(graph, selected_features)


if __name__ == "__main__":
    main()
