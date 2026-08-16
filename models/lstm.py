import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class LSTM(nn.Module):
    """
    Enhanced LSTM baseline for multi‑step node forecasting.

    Input:  (B, in_dim, N, T)
    Output: (B, horizon, N, out_dim)

    Modes:
      - projection (default): uses pooled recent hidden states
      - autoregressive: GRU decoder unrolled over horizon
    """

    def __init__(self, args,
                 node_cnt: int = 1,
                 horizon: int = 12,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 out_dim: int = 1,
                 layer_norm: bool = False,
                 autoregressive: bool = True,
                 decoder_hidden_size: Optional[int] = None,
                ):
        super().__init__()
        self.in_dim = node_cnt
        self.hidden_sizes = args.get("hidden_sizes")
        self.num_layers = len(self.hidden_sizes)
        self.horizon = horizon
        self.out_dim = out_dim
        self.autoregressive = autoregressive
        self.num_directions = 2 if bidirectional else 1
        # embedding dim is based on last layer hidden size
        self.emb_dim = self.num_directions * self.hidden_sizes[-1]
        dec_h = decoder_hidden_size or self.emb_dim

        self.lstm_layers = nn.ModuleList()
        self.layer_dropouts = nn.ModuleList()
        prev_dim = node_cnt
        for idx, h in enumerate(self.hidden_sizes):
            in_sz = prev_dim
            # if previous layer was bidirectional, its output size becomes prev_dim * num_directions
            if idx > 0:
                in_sz = self.hidden_sizes[idx - 1] * self.num_directions
            lstm_layer = nn.LSTM(
                input_size=in_sz,
                hidden_size=h,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.lstm_layers.append(lstm_layer)
            self.layer_dropouts.append(nn.Dropout(dropout if idx < self.num_layers - 1 else 0.0))
            prev_dim = h

        # Apply LN over embedding dim (per timestep), not over flattened context
        self.use_layer_norm = bool(layer_norm)
        self.ln_emb = nn.LayerNorm(self.emb_dim) if self.use_layer_norm else nn.Identity()
        self.rep_dropout = nn.Dropout(dropout)


        if self.autoregressive:
            self.decoder_gru = nn.GRU(
                input_size=self.emb_dim,
                hidden_size=dec_h,
                num_layers=1,
                batch_first=True
            )
            self.dec_proj = nn.Linear(dec_h, out_dim)
            self.init_context = nn.Linear(self.emb_dim, self.emb_dim)
            self.h2in = nn.Identity() if dec_h == self.emb_dim else nn.Linear(dec_h, self.emb_dim)

        self._init_weights()

    def _init_weights(self):
        for lstm in self.lstm_layers:
            for name, p in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
        if self.autoregressive:
            for name, p in self.decoder_gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
            nn.init.xavier_uniform_(self.dec_proj.weight)
            nn.init.zeros_(self.dec_proj.bias)
            nn.init.xavier_uniform_(self.init_context.weight)
            nn.init.zeros_(self.init_context.bias)
            if not isinstance(self.h2in, nn.Identity):
                nn.init.xavier_uniform_(self.h2in.weight)
                nn.init.zeros_(self.h2in.bias)


    def _decode_autoregressive(self,
                               h_last: torch.Tensor,
                               h_seq: torch.Tensor,
                               B: int,
                               N: int) -> torch.Tensor:
        # h_seq: (B*N, T_enc, E)
        mean_ctx = h_seq.mean(dim=1)
        ctx = torch.tanh(self.init_context(mean_ctx + h_last))  # (B*N, E)
        dec_in = ctx.unsqueeze(1)                               # (B*N,1,E)
        hidden = ctx.unsqueeze(0)                               # (1,B*N, E or dec_h)
        preds = []
        for _ in range(self.horizon):
            out_step, hidden = self.decoder_gru(dec_in, hidden)   # out_step: (B*N,1,Hd)
            step = self.dec_proj(out_step[:, -1, :])              # (B*N,out_dim)
            preds.append(step)
            dec_in = self.h2in(out_step)                          # (B*N,1,E)
        pred = torch.stack(preds, dim=1)                        # (B*N,H,out_dim)
        pred = pred.view(B, N, self.horizon, self.out_dim).permute(0, 2, 1, 3).contiguous()
        return pred

    def forward(self,
                x: torch.Tensor,
                hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        x: (B, in_dim, N, T)
        returns: (B, 1, N, H) for out_dim == 1
        """
        assert self.out_dim == 1, "Classification expects out_dim=1"
        assert x.dim() == 4, f"expected 4D input, got {tuple(x.shape)}"
        B, F, N, T = x.shape
        assert F == self.in_dim, f"in_dim mismatch in model: {F} != {self.in_dim}"

        x = x.permute(0, 2, 3, 1).contiguous()   # (B,N,T,F)
        x = x.view(B * N, T, F)                  # (B*N,T,F)

        seq = x
        last_h = None
        last_c = None
        for idx, lstm in enumerate(self.lstm_layers):
            seq, (h_n, c_n) = lstm(seq)  # seq: (B*N, T, H_i * num_directions)
            if idx < len(self.layer_dropouts):
                seq = self.layer_dropouts[idx](seq)
            last_h, last_c = h_n, c_n  

        # seq now is the output of the final LSTM layer: (B*N, T, emb_dim)
        # last_h: (1, num_directions, B*N, last_hidden)
        self._last_h, self._last_c = last_h, last_c
        last = last_h.view(self.num_directions, B * N, self.hidden_sizes[-1]).transpose(0, 1).reshape(B * N, -1)
        self._last_embedding = last.view(B, N, -1)

        if self.autoregressive:
            pred = self._decode_autoregressive(last, seq, B, N)   # (B,H,N,1)
        
        out = pred.permute(0, 3, 2, 1).contiguous()                   # (B,1,N,H)
        assert out.shape == (B, 1, N, self.horizon), f"model output shape mismatch: {tuple(out.shape)}"
        return out