import networkx as nx

def load_graph_from_gml(file_path):
    graph = nx.read_gml(file_path)
    return graph

def main():
    file_path = 'graph_data.gml'
    graph = load_graph_from_gml(file_path)
    print("Nodes:", graph.nodes(data=True))
    print("Edges:", graph.edges(data=True))

if __name__ == "__main__":
    main()