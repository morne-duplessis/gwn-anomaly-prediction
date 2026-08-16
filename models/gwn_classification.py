import torch
import torch.nn as nn
import torch.nn.functional as F

class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNetClassifier(nn.Module):
    def __init__(self, device, node_cnt, dropout=0.3, supports=None, gcn_bool=True, adapt_adj=True, adj_init=None,
                 in_dim=1, out_dim=12, hidden_sizes=None, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, num_classes=None, classification_type='graph'):
        super(GraphWaveNetClassifier, self).__init__()

        # If a swept hidden_sizes value is provided, derive channel widths from it
        # (mirrors the channels -> channels*8 -> channels*16 scaling from the config).
        if hidden_sizes is not None:
            ch = hidden_sizes[-1] if isinstance(hidden_sizes, (list, tuple)) else int(hidden_sizes)
            residual_channels = dilation_channels = ch
            skip_channels = ch * 8
            end_channels = ch * 16

        print(
            f"[GWN CONFIG] hidden_sizes={hidden_sizes} | "
            f"residual_channels={residual_channels} | dilation_channels={dilation_channels} | "
            f"skip_channels={skip_channels} | end_channels={end_channels}"
        )

        # Sweep sanity check: the ConfigSpace sweep injects single-width choices
        # (e.g. [32], [64], ...). A multi-element hidden_sizes such as the argparse
        # default [250, 150] means the swept value never reached the constructor
        # (usually a stale process running pre-sweep code). Fail loudly.
        if isinstance(hidden_sizes, (list, tuple)) and len(hidden_sizes) > 1:
            raise ValueError(
                f"[GWN CONFIG] hidden_sizes={hidden_sizes} has >1 element; expected a "
                f"single swept width (e.g. [64]). The ConfigSpace sweep value did not "
                f"reach the model — likely a stale process running pre-sweep code."
            )

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.adapt_adj = adapt_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports
        self.final_adj = None
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and adapt_adj:
            if adj_init is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(node_cnt, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, node_cnt).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(adj_init)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=(1, new_dilation)))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1, new_dilation)))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        # Classification head
        self.num_classes = num_classes


        print("\n========== MODEL SUMMARY ==========")
        print(self)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print("===================================\n")

        # from torchinfo import summary

        # try:
        #     summary(
        #         self,
        #         input_size=(1, 1, node_cnt, 6),  # batch, channels, nodes, window
        #         depth=3
        #     )
        # except Exception as e:
        #     print("Summary failed:", e)
        

    def forward(self, inputs):
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.adapt_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if self.supports is not None:
                new_supports = self.supports + [adp]
            else:
                new_supports = [adp]
            self.final_adj = new_supports

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            filter_ = self.filter_convs[i](residual)
            filter_ = torch.tanh(filter_)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter_ * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except BaseException:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.adapt_adj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # x shape: [batch, out_dim, num_nodes, time]

        # If classification head is present, return logits
        if self.num_classes is not None:
            pooled = x.mean(dim=3)                          # [batch, out_dim=horizon, num_nodes]
            logits = pooled.permute(0, 2, 1).unsqueeze(1)   # [batch, 1, num_nodes, horizon]
            return logits

        # default behaviour: return raw output (as before)
        return x