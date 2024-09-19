import random
import networkx as nx
import numpy as np
from networkx.algorithms.cuts import conductance
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


def split_segments(composer_segments):
    random.shuffle(composer_segments)
    midpoint = len(composer_segments) // 2

    list1 = composer_segments[:midpoint]
    list2 = composer_segments[midpoint:]

    return list1, list2


def graph_metrics_dict(graph: nx.classes.graph.Graph, distance_matrix: np.array, seed):
    dtw_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]

    node_clustering = nx.clustering(graph)

    communities = nx.community.louvain_communities(graph, seed=seed)

    silhouette_scores = []
    for cluster in communities:
        for i in cluster:
            cluster_distances = distance_matrix[i, list(cluster - {i})]
            homogeneity = np.mean(cluster_distances)
            other_cluster_distances = [np.mean(distance_matrix[i, list(other_cluster)]) for other_cluster in communities
                                       if other_cluster != cluster]
            heterogeneity = min(other_cluster_distances) if other_cluster_distances else homogeneity
            silhouette_score = (heterogeneity - homogeneity) / max(heterogeneity, homogeneity)
            silhouette_scores.append(silhouette_score)

    conductance_scores = []
    for cluster in communities:
        cluster_conductance = conductance(graph, cluster)
        conductance_scores.append(cluster_conductance)

    graph_data = dict()
    graph_data['dtw_distance'] = dtw_distances
    graph_data['clustering'] = list(node_clustering.values())
    graph_data['silhouette_score'] = silhouette_scores
    graph_data['conductance'] = conductance_scores

    return graph_data


def create_split_graph_df(s1_gd: dict, s2_gd: dict, metric: str):
    split_graph_df = pd.DataFrame({
        f"{metric}": np.concatenate([s1_gd[f"{metric}"], s2_gd[f"{metric}"]]),
        "group": ['s1'] * len(s1_gd[f"{metric}"]) + ['s2'] * len(s2_gd[f"{metric}"])
        })

    return split_graph_df


def get_graph_anova(split_graph_df: pd.DataFrame, metric: str):
    anova = ols(f'{metric} ~ C(group)', data=split_graph_df).fit()
    anova_table = sm.stats.anova_lm(anova, typ=1)

    return anova, anova_table


def residual_normality_tests(residuals):
    shapiro = stats.shapiro(residuals)
    print(f'Shapiro-Wilk test: p-value = {shapiro.pvalue}')
    print("Is Normal?:", shapiro.pvalue > 0.05)

    # Plot should follow a relatively straight line to indicate normality
    plt.figure()
    sm.qqplot(residuals, line='s')
    plt.title('QQ Plot')
    plt.show()

    # Plot should follow the normal distribution to indicate normality
    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Histogram with KDE')
    plt.show()


def variance_test(split_graph_df: pd.DataFrame, metric: str):
    levene_test = stats.levene(split_graph_df[split_graph_df['group'] == 's1'][f'{metric}'],
                               split_graph_df[split_graph_df['group'] == 's2'][f'{metric}'])
    print(f"Levene Statistic: {levene_test.statistic}")
    print(f"Levene P Value: {levene_test.pvalue}")
    print(f"Variances are significantly different?: {levene_test.pvalue < 0.05}")


def welch_anova_wrapper(split_graph_df: pd.DataFrame, metric: str):
    welch_anova = pg.welch_anova(dv=f'{metric}', between='group', data=split_graph_df)
    print("WELCH ANOVA")
    print(welch_anova)


def kruskal_wallis(split_graph_df: pd.DataFrame, metric: str):
    kruskal = stats.kruskal(split_graph_df[split_graph_df['group']=='s1'][f'{metric}'],
                            split_graph_df[split_graph_df['group']=='s2'][f'{metric}'])
    print("KW Statistic",kruskal.statistic)
    print("KW P-Value", kruskal.pvalue)


def hypothesis_tests(graph_data_1: dict, graph_data_2: dict, metric: str):
    graph_data_df = create_split_graph_df(graph_data_1, graph_data_2, metric)
    anova, anova_table = get_graph_anova(graph_data_df, metric)
    print("==============================")
    print("ANOVA")
    print(anova_table)
    print("==============================")

    print("RESIDUAL NORMALITY TESTS")
    residual_normality_tests(anova.resid)

    print("==============================")
    print("VARIANCE TEST")
    variance_test(graph_data_df, metric)
    print("==============================")

    print("==============================")
    welch_anova_wrapper(graph_data_df, metric)
    print("==============================")

    print("KRUSKAL WALLIS TEST")
    kruskal_wallis(graph_data_df, metric)
    print("==============================")
