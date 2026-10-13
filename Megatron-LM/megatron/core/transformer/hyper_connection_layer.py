"""Native standard mHC/xHC transformer layer for Megatron Core.

The adapter supports DP and EP with TP=PP=CP=1. Sequence
partitioning requires exchanging convolution history and is intentionally
rejected until it has its own parity tests.
"""

import json

import torch
from megatron.core import activation_stats
from megatron.core.transformer.transformer_layer import TransformerLayer

from .hyper_connections import HyperConnection, connection_pair, expand_streams, reduce_streams

MEGATRON_COMMIT = "55ac7082517c3878ae653c07c09c534b8aed49f6"


def validate_config(config):
    for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                 "context_parallel_size", "expert_tensor_parallel_size"):
        if getattr(config, name) != 1:
            raise ValueError(f"HyperConnectionLayer requires {name}=1")
    for name in ("sequence_parallel", "fp8", "fp4", "hidden_dropout",
                 "attention_dropout", "apply_residual_connection_post_layernorm",
                 "fine_grained_activation_offloading", "cpu_offloading"):
        if getattr(config, name, False):
            raise ValueError(f"HyperConnectionLayer does not support {name}")
    if config.hyper_connection not in ('mhc', 'xhc'):
        raise ValueError('HyperConnectionLayer requires mhc or xhc')
    if config.mtp_num_layers or config.multi_latent_attention:
        raise ValueError('HC integration currently supports GQA/MHA without MTP or MLA')
    if config.cuda_graph_impl != "none" or config.recompute_granularity not in (None, "full"):
        raise ValueError("Only ordinary execution or full layer recomputation is supported")


class HyperConnectionLayer(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_config(self.config)
        connection_kind = self.config.hyper_connection
        connection_seed = self.config.hyper_connection_seed
        connection_settings = json.loads(self.config.hyper_connection_settings or '{}')
        if not isinstance(connection_settings, dict):
            raise ValueError('hyper_connection_settings must be a JSON object')
        if connection_kind == 'xhc':
            connection_settings.setdefault('one_dim_post', True)
        options = {"init_std": self.config.init_method_std, **(connection_settings or {})}
        options["backend"] = self.config.hyper_connection_backend
        attn, mlp = connection_pair(self.config.hidden_size, connection_kind, options)
        self.streams = attn.streams
        # Preserve the backbone RNG stream so all three models get identical
        # common weights under the same seed. Connection modules initialize on CPU.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(connection_seed + self.layer_number)
            self.attn_connection = HyperConnection(attn)
            self.mlp_connection = HyperConnection(mlp)

    @staticmethod
    def _with_bias(result):
        output, bias = result
        return output if bias is None else output + bias

    def forward(self, hidden_states, attention_mask=None, context=None,
                context_mask=None, rotary_pos_emb=None, rotary_pos_cos=None,
                rotary_pos_sin=None, rotary_pos_cos_sin=None, attention_bias=None,
                inference_context=None, packed_seq_params=None, sequence_len_offset=None,
                padding_mask=None, *, inference_params=None):
        if any(value is not None for value in (context, context_mask, inference_context,
                                               inference_params, packed_seq_params)):
            raise ValueError("Only unpacked full-sequence decoder training is supported")
        if self.layer_number == 1:
            hidden_states = expand_streams(hidden_states, self.streams)
        single, state = self.attn_connection.read(hidden_states)
        activation_stats.record('attn_input', self.layer_number, single)
        result = self.self_attention(
            self.input_layernorm(single), attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb, rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin, rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias, sequence_len_offset=sequence_len_offset,
        )
        activation_stats.record('attn_output', self.layer_number, result[0])
        hidden_states = self.attn_connection.write(self._with_bias(result), state)
        activation_stats.record('post_attn_residual_output', self.layer_number, hidden_states)
        single, state = self.mlp_connection.read(hidden_states)
        activation_stats.record('mlp_input', self.layer_number, single)
        result = self.mlp(self.pre_mlp_layernorm(single), padding_mask=padding_mask)
        activation_stats.record('mlp_output', self.layer_number, result[0])
        hidden_states = self.mlp_connection.write(self._with_bias(result), state)
        activation_stats.record('post_mlp_residual_output', self.layer_number, hidden_states)
        if self.layer_number == self.config.num_layers:
            hidden_states = reduce_streams(hidden_states)
        return hidden_states, None
