import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load the GML file
gml_file = 'youtube_cyberbullying_graph.gml'
G = nx.read_gml(gml_file)

# Print basic information about the graph
print("Graph Info:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is directed: {G.is_directed()}")

# Compute centrality measures
pr = nx.pagerank(G, alpha=0.8)
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Compute eigenvector centrality with increased iterations and tolerance
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
except nx.PowerIterationFailedConvergence as e:
    print("\nEigenvector centrality failed to converge. Consider further increasing 'max_iter' or 'tol'.")
    eigenvector_centrality = {}

# Function to print the top N nodes for a given centrality measure
def print_top_nodes(centrality_name, centrality_dict, n=10):
    print(f"\nTop {n} nodes by {centrality_name}:")
    top_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    for node, centrality in top_nodes:
        print(f"Node: {node}, {centrality_name}: {centrality:.4f}")
    return top_nodes

# Function to plot top nodes for a given centrality measure
def plot_top_nodes(centrality_name, top_nodes):
    nodes, values = zip(*top_nodes)
    plt.figure(figsize=(10, 6))
    plt.bar(nodes, values, color='skyblue')
    plt.xlabel("Node")
    plt.ylabel(centrality_name)
    plt.title(f"Top Nodes by {centrality_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Print and plot the top 10 nodes for each centrality measure
top_pagerank = print_top_nodes("PageRank", pr, 10)
plot_top_nodes("PageRank", top_pagerank)

top_degree = print_top_nodes("Degree Centrality", degree_centrality, 10)
plot_top_nodes("Degree Centrality", top_degree)

top_closeness = print_top_nodes("Closeness Centrality", closeness_centrality, 10)
plot_top_nodes("Closeness Centrality", top_closeness)

top_betweenness = print_top_nodes("Betweenness Centrality", betweenness_centrality, 10)
plot_top_nodes("Betweenness Centrality", top_betweenness)

if eigenvector_centrality:
    top_eigenvector = print_top_nodes("Eigenvector Centrality", eigenvector_centrality, 10)
    plot_top_nodes("Eigenvector Centrality", top_eigenvector)

# Save the centrality measures to a CSV file
centrality_data = {
    "Node": list(pr.keys()),
    "PageRank": list(pr.values()),
    "Degree Centrality": list(degree_centrality.values()),
    "Closeness Centrality": list(closeness_centrality.values()),
    "Betweenness Centrality": list(betweenness_centrality.values()),
}
if eigenvector_centrality:
    centrality_data["Eigenvector Centrality"] = list(eigenvector_centrality.values())

centrality_df = pd.DataFrame(centrality_data)
centrality_df.to_csv('graph_centrality_measures.csv', index=False)
print("\nCentrality measures saved to 'graph_centrality_measures.csv'")
