import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(graph):
        nx_g = graph.to_networkx().to_undirected()
        pos = nx.spring_layout(nx_g)  # Layout for visualization
        plt.figure(figsize=(8, 8))
        nx.draw(nx_g, pos, with_labels=True, node_size=500, node_color="skyblue", alpha=0.7, edge_color="gray")
        plt.show()
        
