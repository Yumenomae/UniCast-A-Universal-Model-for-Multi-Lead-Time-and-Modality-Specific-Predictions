# Third-party
import torch
import torch.nn as nn

# First-party
from src.models.components.GraphCast import utils
from src.models.components.GraphCast.interaction_net import InteractionNet, PropagationNet


class BaseGraphModel(nn.Module):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    def __init__(
            self,
            graph,
            hidden_dim,
            hidden_layers,
            grid_dim,
            ):
        super().__init__()

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        self.hierarchical, graph_ldict = utils.load_graph(graph)
        self.num_grid_nodes = 441 * 845
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.grid_embedder = utils.make_mlp(
            [grid_dim] + self.mlp_blueprint_end
        )
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        gnn_class = PropagationNet
        # encoder
        self.g2m_gnn = gnn_class(
            self.g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [hidden_dim] + self.mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = gnn_class(
            self.m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1)
            + [grid_dim],
            layer_norm=False,
        )  # No layer norm on this one

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x, hour_of_year):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        batch_size, V, H, W = x.shape
        x = x.flatten(2, 3).permute(0, 2, 1)

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = x

        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        # Residual connection for full state

        x = (x + net_output).reshape(batch_size, H, W, V)
        return x.permute(0, 3, 1, 2)