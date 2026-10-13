"""CE-only global token mean for one-microbatch DP training.

The ordinary schedule divides CE by local_tokens and DP averages gradients.
Rescaling only CE by local_tokens * dp_size / global_tokens gives exactly
sum(CE) / sum(valid_tokens), while leaving auxiliary autograd losses unchanged.
"""
import torch


def rescale_ce_for_global_tokens(loss, local_tokens, dp_group, num_microbatches):
    if num_microbatches != 1:
        raise ValueError("global-token-mean-loss requires one microbatch per optimizer step")
    total_tokens = local_tokens.detach().clone().to(dtype=torch.float32)
    torch.distributed.all_reduce(total_tokens, group=dp_group)
    if total_tokens.item() <= 0:
        raise ValueError("The global batch has no valid target tokens")
    dp_size = torch.distributed.get_world_size(dp_group)
    return loss * (local_tokens.float() * dp_size / total_tokens)
