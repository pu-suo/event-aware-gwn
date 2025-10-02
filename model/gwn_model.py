import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    """Graph conv operation."""
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        Args:
            x (torch.Tensor): Input data of shape (B, C, N, T).
            A (torch.Tensor): Adjacency matrix of shape (B, T, N, N) for dynamic graphs
                              or (N, N) for static graphs.
        Returns:
            torch.Tensor: Output data of shape (B, C, N, T).
        """
        if len(A.shape) == 4: # Dynamic, time-varying graph
            x = x.permute(0, 1, 3, 2)
            ax = torch.einsum('bctn,btmn->bctm', x, A)
            return ax.permute(0, 1, 3, 2)
        else: # Static graph
            ax = torch.einsum('bcnl,nm->bcml', x, A)
            return ax

class DynamicGraphAttention(nn.Module):
    """
    Computes a dynamic, event-driven adjacency matrix using cross-attention.
    """
    def __init__(self, event_feature_dim, node_embedding_dim, d_k):
        super(DynamicGraphAttention, self).__init__()
        self.d_k = d_k
        self.w_q = nn.Linear(event_feature_dim, d_k, bias=False)
        self.w_k = nn.Linear(node_embedding_dim, d_k, bias=False)

    def forward(self, event_features, node_embeddings):
        """
        Args:
            event_features (torch.Tensor): Shape (B, T, N, F).
            node_embeddings (torch.Tensor): Shape (N, d).
        Returns:
            torch.Tensor: Event-driven attention matrix A_event of shape (B, T, N, N).
        """
        queries = self.w_q(event_features)
        keys = self.w_k(node_embeddings)
        scores = torch.einsum('btnd,md->btnm', queries, keys)
        scaled_scores = scores / (self.d_k ** 0.5)
        A_event = F.softmax(scaled_scores, dim=-1)
        return A_event

class SpatioTemporalBlock(nn.Module):
    """
    A robust Spatio-Temporal block with gated TCN and dynamic graph convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, event_feature_dim, node_embedding_dim, d_k):
        super(SpatioTemporalBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation
        
        self.filter_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), dilation=dilation)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), dilation=dilation)
        
        self.dga_module = DynamicGraphAttention(event_feature_dim, node_embedding_dim, d_k)
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.nconv = nconv()
        
        self.residual_conv = nn.Conv2d(out_channels, in_channels, kernel_size=(1, 1))
        self.skip_conv = nn.Conv2d(out_channels, 32, kernel_size=(1, 1)) 

    def forward(self, traffic_x, event_x, node_embeddings, A_static, A_adaptive):
        residual = traffic_x
        x_padded = F.pad(traffic_x, (self.padding, 0))
        
        filter_out = torch.tanh(self.filter_conv(x_padded))
        gate_out = torch.sigmoid(self.gate_conv(x_padded))
        x_temporal = filter_out * gate_out
        
        A_event = self.dga_module(event_x, node_embeddings)
        A_final = A_static + A_adaptive + self.alpha * A_event
        
        x_gcn = self.nconv(x_temporal, A_final)
        
        skip_out = self.skip_conv(x_gcn)
        res_out = self.residual_conv(x_gcn)
        res_out = res_out + residual

        return res_out, skip_out

class AttentionModulatedGWN(nn.Module):
    def __init__(self, supports, num_nodes, in_channels, out_channels, seq_length, 
                 event_feature_dim, node_embedding_dim, d_k, dropout=0.3, blocks=4, layers=2):
        super(AttentionModulatedGWN, self).__init__()
        
        residual_channels = 32
        dilation_channels = 32
        end_channels = 512

        self.E1 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim), requires_grad=True)
        self.E2 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim), requires_grad=True)
        self.register_buffer('supports', torch.from_numpy(supports[0]).float())
        
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=residual_channels, kernel_size=(1,1))
        
        self.st_blocks = nn.ModuleList()
        
        for b in range(blocks):
            for l in range(layers):
                dilation_factor = 2**l
                self.st_blocks.append(SpatioTemporalBlock(
                    in_channels=residual_channels, out_channels=dilation_channels, 
                    kernel_size=2, dilation=dilation_factor,
                    event_feature_dim=event_feature_dim,
                    node_embedding_dim=node_embedding_dim, d_k=d_k
                ))
        
        self.end_conv_1 = nn.Conv2d(in_channels=32 * blocks * layers, out_channels=end_channels, kernel_size=(1,1))
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1,1))

    def forward(self, traffic_input, event_input):
        x = traffic_input.permute(0, 3, 2, 1)
        A_adaptive = F.softmax(F.relu(torch.mm(self.E1, self.E2.T)), dim=1)
        A_static = self.supports

        x = self.start_conv(x)
        skip_connections = []
        
        for block in self.st_blocks:
            x, skip_connection = block(x, event_input, self.E1, A_static, A_adaptive)
            skip_connections.append(skip_connection)
        
        skip_cat = torch.cat(skip_connections, dim=1)
        
        skip_processed = skip_cat[..., -1:]
        
        x = F.relu(self.end_conv_1(skip_processed))
        x = self.end_conv_2(x)
        
        return x.squeeze(3)

