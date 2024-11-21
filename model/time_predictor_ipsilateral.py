import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, batch_first=True
        )

    def forward(self, query, key, value):
        # Perform cross-attention between query (previous swing phase) and key/value (current swing phase)
        attn_output, _ = self.attn(query, key, value)
        return attn_output


class AdjustedGaitModel(nn.Module):
    def __init__(self, input_size, embed_size=128, num_heads=2, num_layers=2, dropout=0.1):
        super(AdjustedGaitModel, self).__init__()

        self.embed_size = embed_size

        # Embedding layers for input sequences (previous swing phase and current swing phase)
        self.fc_input = nn.Linear(input_size, embed_size)

        # Cross-attention layer for previous and current swing phases
        self.cross_attention = CrossAttentionLayer(embed_size, num_heads)

        # More fully connected layers to increase model complexity
        self.fc_layers = nn.ModuleList(
            [nn.Linear(embed_size, embed_size) for _ in range(num_layers)]
        )

        # Add Batch Normalization for better gradient flow
        self.batch_norm = nn.BatchNorm1d(embed_size)

        # Output layer to predict stride time
        self.fc_output = nn.Linear(embed_size, 1)

    def forward(self, prev_swing_phase, current_swing_phase):
        """
        prev_swing_phase: Tensor of shape [batch_size, seq_len, input_size]
        current_swing_phase: Tensor of shape [batch_size, seq_len, input_size]
        """

        # Embed the input sequences
        prev_swing_embedded = F.relu(self.fc_input(prev_swing_phase))  # Shape: [batch_size, seq_len, embed_size]
        current_swing_embedded = F.relu(self.fc_input(current_swing_phase))  # Shape: [batch_size, seq_len, embed_size]

        # Apply cross-attention between previous and current swing phases
        attn_output = self.cross_attention(prev_swing_embedded, current_swing_embedded, current_swing_embedded)

        # Process through fully connected layers
        output = attn_output
        for layer in self.fc_layers:
            output = F.relu(layer(output))

        # Apply Batch Normalization after fully connected layers
        output = output.view(-1, self.embed_size)  # Flatten the tensor for BatchNorm1d
        output = self.batch_norm(output)  # Apply Batch Normalization

        # Revert to the original shape
        output = output.view(prev_swing_phase.shape[0], prev_swing_phase.shape[1], self.embed_size)

        # Output prediction (stride time)
        stride_time = self.fc_output(output)  # Shape: [batch_size, seq_len, 1]

        # Use the last stride time prediction
        stride_time = stride_time[:, -1]

        return stride_time
