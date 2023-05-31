import torch
import torch.nn as nn
from models.modules.lap_pyramid import Lap_Pyramid_Bicubic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_num #3
        self.split_len2 = channel_split_num #12-3
        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class InvBlockExpTail(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExpTail, self).__init__()

        self.split_len1 = channel_num #3
        self.split_len2 = channel_split_num # 12-3

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        x1 = x
        x2 = torch.cat([x,x,x],1)
        print(x1.shape, x2.shape)
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)
        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class InvBlockExp_RNVP(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1., clamp_activation='ATAN'):
        super(InvBlockExp_RNVP, self).__init__()

        self.split_len1 = channel_num #3
        self.split_len2 = channel_split_num # 12-3

        self.clamp = clamp

        self.subnet_s1 = subnet_constructor(self.split_len1, self.split_len2)
        self.subnet_t1 = subnet_constructor(self.split_len1, self.split_len2)
        self.subnet_s2 = subnet_constructor(self.split_len2, self.split_len1)
        self.subnet_t2 = subnet_constructor(self.split_len2, self.split_len1)

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = (lambda u: 2. * (torch.sigmoid(u) - 0.5))
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))


        if not rev:
            s2, t2 = self.subnet_s2(x2), self.subnet_t2(x2)
            s2 = self.clamp * self.f_clamp(s2)
            y1 = torch.exp(s2) * x1 + t2

            s1, t1 = self.subnet_s1(y1), self.subnet_t1(y1)
            s1 = self.clamp * self.f_clamp(s1)
            y2 = torch.exp(s1) * x2 + t1

        else:
            s1, t1 = self.subnet_s1(x1), self.subnet_t1(x1)
            s1 = self.clamp * self.f_clamp(s1)
            y2 = (x2-t1)*torch.exp(-s1)

            s2, t2 = self.subnet_s2(y2), self.subnet_t2(y2)
            s2 = self.clamp * self.f_clamp(s2)
            y1 = (x1-t2)*torch.exp(-s2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class InvBlockExp_RNVP_SGT(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, cond_channel, clamp=1., clamp_activation='ATAN'):
        super(InvBlockExp_RNVP_SGT, self).__init__()

        self.split_len1 = channel_num #3
        self.split_len2 = channel_split_num # 12-3
        self.cond_channel = cond_channel # 12-3

        self.clamp = clamp

        self.subnet_s1 = subnet_constructor(self.split_len1, self.split_len2, self.cond_channel)
        self.subnet_t1 = subnet_constructor(self.split_len1, self.split_len2, self.cond_channel)
        self.subnet_s2 = subnet_constructor(self.split_len2, self.split_len1, self.cond_channel)
        self.subnet_t2 = subnet_constructor(self.split_len2, self.split_len1, self.cond_channel)

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = (lambda u: 2. * (torch.sigmoid(u) - 0.5))
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, cond,  rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            s2, t2 = self.subnet_s2(x2, cond), self.subnet_t2(x2, cond)
            s2 = self.clamp * self.f_clamp(s2)
            y1 = torch.exp(s2) * x1 + t2

            s1, t1 = self.subnet_s1(y1, cond), self.subnet_t1(y1, cond)
            s1 = self.clamp * self.f_clamp(s1)
            y2 = torch.exp(s1) * x2 + t1

        else:
            s1, t1 = self.subnet_s1(x1, cond), self.subnet_t1(x1, cond)
            s1 = self.clamp * self.f_clamp(s1)
            y2 = (x2-t1)*torch.exp(-s1)

            s2, t2 = self.subnet_s2(y2, cond), self.subnet_t2(y2, cond)
            s2 = self.clamp * self.f_clamp(s2)
            y1 = (x1-t2)*torch.exp(-s2)

        return torch.cat((y1, y2), 1)




def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x

def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = squeeze2d(input, self.factor)  # Squeeze in forward
            return output
        else:
            output = unsqueeze2d(input, self.factor)
            return output

class HaarDownsampling(nn.Module):
    def __init__(self, factor):
        super(HaarDownsampling, self).__init__()
        self.squeeze = SqueezeLayer(factor=factor)
    def forward(self, x, rev=False):
        if not rev:
            out = self.squeeze(x)
            return out
        else:
            out = self.squeeze(x, reverse=True)
            return out

class InvNet(nn.Module):
    def __init__(self, subnet_constructor=None, subnet_constructor_sgt=None,  down_num=1):
        super(InvNet, self).__init__()

        self.lap_pyramid = Lap_Pyramid_Bicubic(1)

        operations = []
        for i in range(1):
            b = HaarDownsampling(4)
            operations.append(b)
        self.down_HF = nn.ModuleList(operations)

        operations = []
        for i in range(1):
            b = HaarDownsampling(2)
            operations.append(b)
            for j in range(8):
                b = InvBlockExp_RNVP(subnet_constructor, 8, 4)
                operations.append(b)
        self.operations1 = nn.ModuleList(operations)

        operations = []
        for i in range(down_num):
            for j in range(4):
                b = InvBlockExp_RNVP(subnet_constructor, 8, 48)
                operations.append(b)
        self.operations3_1 = nn.ModuleList(operations)

        operations = []
        for i in range(down_num):
            for j in range(4):
                b = InvBlockExp_RNVP_SGT(subnet_constructor_sgt, 8, 48, 4)
                operations.append(b)
        self.operations3_2 = nn.ModuleList(operations)

        operations = []
        for i in range(down_num):
            for j in range(4):
                b = InvBlockExp_RNVP(subnet_constructor, 8, 48)
                operations.append(b)
        self.operations3_3 = nn.ModuleList(operations)

        operations = []
        for i in range(down_num):
            for j in range(12):
                b = InvBlockExp_RNVP(subnet_constructor, 8, 4)
                operations.append(b)
        self.operations2 = nn.ModuleList(operations)

        self.toSDR = HaarDownsampling(2)

        self.projector_s = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        self.projector_oc = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        self.projector_op = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(device)

    def forward(self, x, rev=False):
        out = x

        # forward process
        if not rev:
            # spatial decomposition
            out = self.lap_pyramid.pyramid_decom(img=out)
            HF = out[0]
            LF = out[1]

            HF_ori = HF
            LF_ori = LF
            out = HF
            for op in self.down_HF:
                out = op.forward(out, rev)
            HF = out

            # strcuture decomposition
            out = LF
            for op in self.operations1:
                out = op.forward(out, rev)
            LF_out = out
            p1 = LF_out[:,0:8,:,:]
            c1 = LF_out[:,8:12,:,:]

            # Translation
            inn3_input = torch.cat([p1, HF],1)
            out = inn3_input
            for op in self.operations3_1:
                out = op.forward(out, rev)

            for op in self.operations3_2:
                out = op.forward(out, c1, rev)

            for op in self.operations3_3:
                out = op.forward(out, rev)
            inn3_output = out

            p2 = inn3_output[:,:8,:,:]

            # synthesis
            inn2_input = torch.cat([p2, c1], dim=1)
            out = inn2_input
            for op in self.operations2:
                out = op.forward(out, rev)
            inn2_output = out

            out = self.toSDR(inn2_output, rev=True)

            proj_s = self.projector_s(c1)
            proj_oc = self.projector_oc(c1)
            proj_op =self.projector_op(p1)

            return out, p1, c1, proj_s, proj_oc, proj_op, HF_ori, LF_ori

        # reverse process
        else:
            LF_ori = out
            out = self.toSDR(out, rev=False)

            for op in reversed(self.operations2):
                out = op.forward(out, rev)

            p2 = out[:,0:8,:,:]
            c2 = out[:,8::,:,:]

            SDR_imgnoise = self.gaussian_batch(
                [p2.shape[0], 48, p2.shape[2], p2.shape[3]])

            out = torch.cat([p2, SDR_imgnoise], 1)
            for op in reversed(self.operations3_3):
                out = op.forward(out, rev)

            for op in reversed(self.operations3_2):
                out = op.forward(out,c2, rev)

            for op in reversed(self.operations3_1):
                out = op.forward(out, rev)
            HF = out[:,8:,:,:]
            p1 = out[:,:8,:,:]

            out = torch.cat([p1,c2], dim=1)
            for op in reversed(self.operations1):
                out = op.forward(out, rev)
            LF = out

            out = HF
            for op in reversed(self.down_HF):
                out = op.forward(out, rev)
            HF = out

            out = self.lap_pyramid.pyramid_recons([HF, LF])

            proj_s = self.projector_s(c2)
            proj_oc = self.projector_oc(c2)
            proj_op = self.projector_op(p2)

            return out, p2, c2, proj_s, proj_oc, proj_op, HF, LF_ori


if __name__ =='__main__':
    from models.modules.Subnet_constructor import subnet
    HDR = torch.ones(2, 3, 160, 160)
    SDR = torch.ones(1, 3, 80, 80)
    p1 = torch.randn(1, 8, 40, 40)
    c1 = torch.randn(1, 4, 40, 40)
    p2 = torch.randn(1,  8, 40, 40)
    c2 = torch.randn(1, 4, 40, 40)
    HF = torch.randn(1, 3, 160, 160)
    model = InvNet(subnet('ResSENet'), subnet('AFF1'), 1)

    out, p1, c1, proj_s, proj_oc, proj_op, _, _ = model(HDR, False)

    print(out.shape, p1.shape, c1.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params/1000**2)
    from thop.profile import profile

    name = "our"
    total_ops, total_params = profile(model, (HDR, ))
    print("%s         | %.4f(M)      | %.4f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))



