"""Qwen linear attention with Megatron's tensor/context-parallel GDN."""

import copy

import torch
import torch.nn.functional as F
from fla.modules.fused_norm_gate import rms_norm_gated
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet


class Qwen3_5NativeAttention(GatedDeltaNet):
    def _prepare_input_for_gated_delta_rule(self, qkv, gate, batch, seq_len, *gate_feats):
        result = super()._prepare_input_for_gated_delta_rule(qkv, gate, batch, seq_len, *gate_feats)
        trace = getattr(self, "_coding_rl_replay_trace", None)
        if trace is not None:
            trace.record("components.gdn_convolution", qkv.reshape(-1, qkv.shape[-1]))
            for name, value in zip(("query", "key", "value"), result[:3]):
                trace.record(f"components.gdn_{name}", value.reshape(-1, value.shape[-2] * value.shape[-1]))
        return result

    def _apply_gated_norm(self, x, gate):
        # Preserve the FP32 norm/scale/gate intermediates used by Qwen's FLA
        # fast path and SGLang. A standalone BF16 norm rounds too early.
        # Reuse the original parameter, including its TP gradient metadata.
        trace = getattr(self, "_coding_rl_replay_trace", None)
        if trace is not None:
            trace.record("components.gdn_raw", x.reshape(-1, x.shape[-2] * x.shape[-1]))
            trace.record("components.gdn_gate", gate.reshape(-1, gate.shape[-2] * gate.shape[-1]))
        return rms_norm_gated(
            x.reshape(-1, x.shape[-1]),
            gate.reshape(-1, gate.shape[-1]),
            self.out_norm.weight,
            None,
            activation="swish",
            eps=self.config.layernorm_epsilon,
        )

    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        # Qwen evaluates the decay exponent in FP32 even with BF16 weights.
        # MCore's default exp on the BF16 parameter rounds before multiplying
        # the FP32 softplus, which accumulates over recurrent layers.
        g = -A_log_local_cp.float().exp() * F.softplus(alpha.float() + dt_bias_local_cp.float())
        beta = beta.sigmoid()
        trace = getattr(self, "_coding_rl_replay_trace", None)
        if trace is not None:
            trace.record("components.gdn_decay", g.reshape(-1, g.shape[-1]))
            trace.record("components.gdn_beta", beta.reshape(-1, beta.shape[-1]))
        return g, beta

    def _resolve_cu_seqlens(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size=1):
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        # Every layer and its recomputation share the same immutable batch
        # offsets. Cache successful validation on that tensor, not the model.
        # Include its version so in-place edits and edits through views recheck.
        if cu_seqlens.is_inference():
            return super()._resolve_cu_seqlens(cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size)
        key = (cu_seqlens._version, total_seq_len, cp_size)
        if getattr(cu_seqlens, "_slime_gdn_validation", None) != key:
            super()._resolve_cu_seqlens(cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size)
            cu_seqlens._slime_gdn_validation = key
        return cu_seqlens

    def __init__(self, config, linear_config, **kwargs):
        config = copy.copy(config)
        for name, value in linear_config.items():
            setattr(config, name, value)
        if not config.layernorm_zero_centered_gamma:
            raise ValueError("Native Qwen GDN requires --apply-layernorm-1p for its input layernorm")
        if config.tensor_model_parallel_size > 1 and not config.sequence_parallel:
            raise ValueError("Native Qwen GDN requires --sequence-parallel to synchronize its output-norm gradients")
        super().__init__(config=config, **kwargs)

        # Qwen's input norm uses (1 + weight), while the gated output norm
        # stores its scale directly. Keep the checkpoint values unchanged.
        self.out_norm.zero_centered_gamma = False
        with torch.no_grad():
            self.out_norm.weight.fill_(1)

        # Each TP rank owns a slice of every projection section. These sizes
        # describe the full tensor, before sharding, for checkpoint conversion
        # and live weight publication to the serving model.
        self.in_proj.weight.partition_sections = tuple(size * self.tp_size for size in self.in_proj_split_sections)
        self.conv1d.weight.partition_sections = (self.qk_dim, self.qk_dim, self.v_dim)
