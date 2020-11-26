import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EdgeConvRot(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(EdgeConvRot, self).__init__(aggr='mean', flow="target_to_source")  # "Max" aggregation.
        self.mlp = nn.Sequential(nn.Linear(2 * in_channels + edge_channels, out_channels),
                       nn.ReLU(),
                       nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        W = torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        W = self.mlp(W)
        return W

    def propagate(self, edge_index, size, x, edge_attr):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        edge_out = self.message(x_i, x_j, edge_attr)
        # out = scatter_(self.aggr, edge_out, edge_index[i], dim_size=size[i])
        out = scatter(edge_out, edge_index[i], dim=0, dim_size=size[i], reduce=self.aggr)
        return out, edge_out

class Cleaner(nn.Module):

    def __init__(self, in_node_feat, in_edge_feat):
        super(Cleaner, self).__init__()
        self.no_features = 128
        self.conv1 = EdgeConvRot(in_node_feat, in_edge_feat, self.no_features)
        self.conv2 = EdgeConvRot(self.no_features, self.no_features, self.no_features)
        self.conv3 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv4 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv5 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.conv6 = EdgeConvRot(2 * self.no_features, 2 * self.no_features, self.no_features)
        self.lin1 = nn.Linear(self.no_features, 1)

    def forward(self, node_feat, edge_feat, edge_idx):

        x1, edge_x1 = self.conv1(node_feat, edge_idx, edge_feat)
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)

        x2, edge_x2 = self.conv2(x1, edge_idx, edge_x1)
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_idx, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)

        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_idx, torch.cat([edge_x3, edge_x2], dim=1))
        x4 = F.relu(x4)
        edge_x4 = F.relu(edge_x4)

        x5, edge_x5 = self.conv5(torch.cat([x4, x3], dim=1), edge_idx, torch.cat([edge_x4, edge_x3], dim=1))
        x5 = F.relu(x5)
        edge_x5 = F.relu(edge_x5)

        x6, edge_x6 = self.conv6(torch.cat([x5, x4], dim=1), edge_idx, torch.cat([edge_x5, edge_x4], dim=1))
        x6 = F.relu(x6)
        edge_x6 = F.relu(edge_x6)

        out = self.lin1(edge_x6)

        return out, x6, edge_x6