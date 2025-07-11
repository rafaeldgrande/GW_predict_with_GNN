

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, NNConv


class AngleGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_distances=True, use_angles=True):
        """
        Initializes the AngleGNNLayer with specified input and output channels.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_distances (bool, optional): If True, use distance-based edge features. Defaults to True.
            use_angles (bool, optional): If True, use angle-based features. Defaults to True.
        """
        super().__init__()
        self.use_distances = use_distances
        self.use_angles = use_angles
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.use_distances:
            self.edge_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, in_channels * out_channels)
            )
        if self.use_angles:
            self.angle_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, out_channels)
            )

    def forward(self, x, edge_index, edge_attr, angle_index, angles):
        """
        Perform a forward pass through the AngleGNNLayer.

        Args:
            x (torch.Tensor): Node feature matrix of shape (N, in_channels),
                            where N is the number of nodes.
            edge_index (torch.Tensor): Edge index matrix of shape (2, E), 
                                    where E is the number of edges.
            edge_attr (torch.Tensor): Edge feature matrix of shape (E, 1).
            angle_index (torch.Tensor): Tensor of shape (3, A) indicating 
                                        angle triplets, where A is the number of angles.
            angles (torch.Tensor): Angle feature matrix of shape (A, 1).

        Returns:
            torch.Tensor: Aggregated node features of shape (N, out_channels).
        """
   
        agg = torch.zeros(x.size(0), self.out_channels, device=x.device)

        if self.use_distances:
            row, col = edge_index
            edge_weights = self.edge_mlp(edge_attr).view(-1, self.in_channels, self.out_channels)
            msg = torch.bmm(x[col].unsqueeze(1), edge_weights).squeeze(1)
            agg = agg.index_add(0, row, msg)

        if self.use_angles and angle_index.numel() > 0:
            i, j, k = angle_index
            angle_feat = self.angle_mlp(angles)
            angle_msg = torch.zeros(x.size(0), self.out_channels, device=x.device)
            angle_msg = angle_msg.index_add(0, j, angle_feat)
            agg += angle_msg

        return agg

class AngleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout_rate=0.1,
                 use_distances=True, use_angles=True):
        """
        Initialize the AngleGNN with specified input and output dimensions.

        Args:
            input_dim (int): Input dimensionality of the node features.
            hidden_dims (list): List of hidden layer dimensionalities.
            activation (str, optional): Activation function to use. Defaults to "relu".
            dropout_rate (float, optional): Dropout rate to use. Defaults to 0.1.
            use_distances (bool, optional): If True, use distance-based edge features. Defaults to True.
            use_angles (bool, optional): If True, use angle-based features. Defaults to True.
        """
        super().__init__()
        self.activation = getattr(F, activation)
        self.dropout_rate = dropout_rate
        self.use_distances = use_distances
        self.use_angles = use_angles

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(
            AngleGNNLayer(input_dim, hidden_dims[0], use_distances, use_angles)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        for i in range(1, len(hidden_dims)):
            self.layers.append(
                AngleGNNLayer(hidden_dims[i-1], hidden_dims[i], use_distances, use_angles)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        """
        Forward pass of the AngleGNN model.

        Args:
            data (Data): Input data containing node features, edge indices and attributes,
                angle indices and values, and batch information.

        Returns:
            torch.Tensor: Predicted scalar target property of shape (batch_size,).
        """

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        angle_index = data.angle_index
        angles = data.angles
        batch = data.batch

        for layer, bn in zip(self.layers, self.bns):
            x = layer(x, edge_index, edge_attr, angle_index, angles)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        x = torch.cat([x, data.edft.view(-1, 1)], dim=1)
        return self.mlp(x).squeeze(-1)

class GNNOptunaModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, dropout_rate):
        """
        Initialize the GNNOptunaModel with specified input and hidden layer dimensions, activation function, and dropout rate.

        Args:
            input_dim (int): Input dimensionality.
            hidden_dims (list): List of hidden layer dimensionalities.
            activation (str): Activation function to use. Defaults to "relu".
            dropout_rate (float): Dropout rate to use. Defaults to 0.1.
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activation = getattr(F, activation)
        self.dropout_rate = dropout_rate

        # Define NN for edge features → weight matrices
        def edge_nn(in_channels, out_channels):
            """
            Neural network to map edge features to weight matrices.

            Args:
                in_channels (int): Input dimensionality of node features.
                out_channels (int): Output dimensionality of node features.

            Returns:
                nn.Sequential: Neural network with two hidden layers.
            """
            return nn.Sequential(
                nn.Linear(1, 32),   # edge_attr has shape [num_edges, 1]
                nn.ReLU(),
                nn.Linear(32, in_channels * out_channels)
            )

        # First NNConv layer
        self.convs.append(
            NNConv(input_dim, hidden_dims[0], edge_nn(input_dim, hidden_dims[0]), aggr='mean')
        )
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        # Additional layers
        for i in range(1, len(hidden_dims)):
            self.convs.append(
                NNConv(hidden_dims[i-1], hidden_dims[i], edge_nn(hidden_dims[i-1], hidden_dims[i]), aggr='mean')
            )
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):

        """
        Forward pass of the GNNOptunaModel model.

        Args:
            data (Data): Input data containing node features, edge indices and attributes,
                and batch information.

        Returns:
            torch.Tensor: Predicted scalar target property of shape (batch_size,).
        """

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        x = torch.cat([x, data.edft.view(-1, 1)], dim=1)
        return self.mlp(x).squeeze(-1)

def create_gnn_model_for_hyperparameter_search(trial, input_dim, params, use_distances=True, use_angles=True):

    """
    Create a GNN model based on Optuna trial parameters.

    Args:
        trial: Optuna trial object
        input_dim: Input dimensionality of node features
        params: Dictionary of hyperparameter search space
        use_distances: If True, use distance-based edge features. Defaults to True.
        use_angles: If True, use angle-based features. Defaults to True.

    Returns:
        AngleGNN: GNN model with specified hyperparameters
    """

    n_layers_min = params["n_layers_min"]
    n_layers_max = params["n_layers_max"]
    hidden_dims_min = params["hidden_dims_min"]
    hidden_dims_max = params["hidden_dims_max"]
    hidden_dims_step = params["hidden_dims_step"]
    dropout_rate_min = params["dropout_rate_min"]
    dropout_rate_max = params["dropout_rate_max"]
    
    n_layers = trial.suggest_int("n_layers", n_layers_min, n_layers_max)
    hidden_dim = trial.suggest_int("n_units", hidden_dims_min, hidden_dims_max, step=hidden_dims_step)
    hidden_dims = [hidden_dim] * n_layers  # All layers have the same size
    activation = "relu"
    dropout_rate = trial.suggest_float("dropout_rate", dropout_rate_min, dropout_rate_max)

    return AngleGNN(input_dim, hidden_dims, activation, dropout_rate,
                    use_distances=use_distances, use_angles=use_angles)
    
def create_gnn_model_from_params(params, input_dim):
    n_layers = params["n_layers"]
    hidden_dim = params["n_units"]
    hidden_dims = [hidden_dim] * n_layers
    dropout_rate = params["dropout_rate"]
    activation = "relu"

    return GNNOptunaModel(input_dim, hidden_dims, activation, dropout_rate)
