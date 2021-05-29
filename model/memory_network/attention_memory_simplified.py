import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import base_model

class Attention_Memory_Simplified(base_model):
    def __init__(self, mem_slots, input_size, output_size, key_size, head_size, num_heads=4):
        super(Attention_Memory_Simplified, self).__init__()
        self.mem_slots = mem_slots

        self.mem_size = input_size
        self.input_size = input_size
        self.output_size = output_size

        self.head_size = head_size
        self.num_heads = num_heads

        # query-key-value
        self.query_size = key_size
        self.key_size = key_size
        self.value_size = self.head_size

        self.q_projector = nn.Linear(self.mem_size, self.num_heads * self.query_size)
        self.q_layernorm = nn.LayerNorm([self.num_heads, self.query_size])

        self.kv_projector = nn.Linear(self.mem_size, self.num_heads*(self.key_size + self.value_size))
        self.k_layernorm = nn.LayerNorm([self.num_heads, self.key_size])
        self.v_layernorm = nn.LayerNorm([self.num_heads, self.value_size])

        # MLP for attention
        self.concatnate_mlp = nn.Linear(self.num_heads*self.value_size, self.mem_size)
        self.concatnate_layernorm = nn.LayerNorm([self.mem_size])
        self.attention_output_layernorm = nn.LayerNorm([self.mem_size])

        self.output_mlp = nn.Linear(self.mem_size, self.output_size)
        self.output_layernorm = nn.LayerNorm([self.output_size])

    def multihead_attention(self, input):

        q = self.q_projector(input)
        q_reshape = q.view(q.shape[0], q.shape[1], self.num_heads, self.query_size)
        q_reshape = self.q_layernorm(q_reshape)
        q_transpose = q_reshape.permute(0, 2, 1, 3)

        kv = self.kv_projector(input)
        kv_reshape = kv.view(kv.shape[0], kv.shape[1], self.num_heads, (self.key_size + self.value_size))
        k_reshape, v_reshape = torch.split(kv_reshape, [self.key_size, self.value_size], dim=-1)
        k_reshape = self.k_layernorm(k_reshape)
        v_reshape = self.v_layernorm(v_reshape)
        k_transpose = k_reshape.permute(0, 2, 1, 3)
        v_transpose = v_reshape.permute(0, 2, 1, 3)

        q_transpose *= (self.key_size ** -0.5)
        # make it [B, H, N, N]
        dot_product = torch.matmul(q_transpose, k_transpose.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)
        # output is [B, H, N, V]
        weighted_output = torch.matmul(weights, v_transpose)
        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]=[batch_size,mem_slots,num_head*head_size]

        output_transpose = weighted_output.permute(0, 2, 1, 3).contiguous()
        output_transpose = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        output_transpose = self.concatnate_mlp(output_transpose)
        output_transpose = self.concatnate_layernorm(output_transpose)

        return output_transpose

    def attention_over_memory(self, input, memory):
        # [batch_size, input_size]
        # [batch_size, mem_slot, mem_size]
        input_reshape = input.unsqueeze(dim=1)  #[batch_size,1,mem_size]

        memory_plus_input = torch.cat([memory, input_reshape], dim=1) # [batch_size,mem_slot+1,mem_size]

        attention_output = self.multihead_attention(memory_plus_input)
        attention_output = self.attention_output_layernorm(attention_output+memory_plus_input)

        # MLP + ADD + LN
        output = self.output_mlp(attention_output)
        output = F.gelu(output)
        output = self.output_layernorm(output+attention_output)

        return output
        # [batch_size, mem_slot+1,mem_size]

    def forward(self, input, memory):
        output = self.attention_over_memory(input, memory)
        output = output[:, -1, :]
        return output

