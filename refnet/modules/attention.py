import torch.nn as nn

from einops import rearrange
from ldm.util import exists
from ldm.modules.attention import (
    zero_module,
    checkpoint,
    Normalize,
    CrossAttention,
    MemoryEfficientCrossAttention,
    BasicTransformerBlock,
    XFORMERS_IS_AVAILBLE,
)


class SelfTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, dim_head=64, dropout=0., mlp_ratio=4, checkpoint=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.attn = attn_cls(query_dim=dim, heads=dim//dim_head, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x


"""
    Diffusion Transformer block.
    Paper: Scalable Diffusion Models with Transformers
    Arxiv: https://arxiv.org/abs/2212.09748
"""
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class Transformer(nn.Module):
    transformer_type = {
        "vanilla": BasicTransformerBlock,
    }
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=True, type="vanilla", **kwargs):
        super().__init__()
        transformer_block = self.transformer_type[type]
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [transformer_block(inner_dim, n_heads, d_head,
                               dropout=dropout, context_dim=context_dim[d], checkpoint=use_checkpoint, **kwargs)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, inject=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], inject=inject)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def SpatialTransformer(*args, **kwargs):
    return Transformer(type="vanilla", *args, **kwargs)