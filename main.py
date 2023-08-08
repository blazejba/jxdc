import jax
import jax.numpy as jnp
from jax import random


def generate_er_graph(num_nodes, p, key):
    """
    Generate a random graph using the Erdős-Rényi model.
    
    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - p (float): Probability of edge creation between any pair of nodes.
    - key (random.PRNGKey): JAX random PRNG key.

    Returns:
    - node_features (jnp.array): A (num_nodes,) array representing node features.
    - edges (list[tuple[int]]): List of tuples, each tuple (i, j) represents an 
      edge between node i and node j.
    """
    
    # Sample a matrix of random values between 0 and 1
    rand_matrix = random.uniform(key, (num_nodes, num_nodes))
    
    # Create an adjacency matrix where 1 indicates the presence of an edge and 0 absence
    adj_matrix = jnp.triu((rand_matrix < p).astype(jnp.int32), k=1)
    
    # Convert adjacency matrix to edge list
    edges = [
        (i, j) for i in range(num_nodes)
        for j in range(num_nodes)
        if adj_matrix[i, j] == 1
    ]
    
    # Placeholder node features (currently just identity for simplicity)
    node_features = jnp.zeros(num_nodes)
    
    return node_features, edges

@jax.jit
def greedy_sequential(node_features, edges):
    """
    Greedy Sequential Vertex Coloring algorithm in Jax.
    
    Parameters:
    - node_features (jnp.array):
      A (num_nodes,) array representing initial nodes.

    - edges (list[tuple[int]]): List of tuples, each tuple (i, j) represents an edge 
      between node i and node j.

    Returns:
    - node_features (jnp.array): A (num_nodes,) array representing node colors.
    """
    num_nodes = node_features.shape[0]
    colors = jnp.zeros(num_nodes, dtype=jnp.int32)

    for v in range(num_nodes):
        # Obtain the colors of the neighbors
        neighbor_colors = jnp.array([colors[u] for (u, w) in edges if w == v or u == v])

        # Find the smallest color that hasn't been used by neighbors
        for color in range(1, num_nodes + 1):
            if color not in neighbor_colors:
                colors = jnp.where(jnp.arange(num_nodes) == v, color, colors)
                break

    return colors


key = random.PRNGKey(0)
num_nodes = 10
p = 0.3
node_features, edges = generate_er_graph(num_nodes, p, key)

print(f"{node_features=}")
print(f"{edges=}")

colored_nodes = greedy_sequential(node_features, edges)
num_colors = colored_nodes.unique()
print(f"{colored_nodes=}")
print(f"{num_colors=}")