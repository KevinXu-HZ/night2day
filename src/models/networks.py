import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout: layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            norm_layer(dim)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x): return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, 7, bias=False),
                 norm_layer(ngf), nn.ReLU(True)]
        mult = 1
        for _ in range(2):
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1, bias=False),
                      norm_layer(ngf*mult*2), nn.ReLU(True)]
            mult *= 2
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf*mult, norm_layer)]
        for _ in range(2):
            model += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, 3, 2, 1, 1, bias=False),
                      norm_layer(ngf*mult//2), nn.ReLU(True)]
            mult //= 2
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw, pad = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]
        nf = 1
        for n in range(1, n_layers):
            nf_prev = nf; nf = min(2**n, 8)
            seq += [nn.Conv2d(ndf*nf_prev, ndf*nf, kw, 2, pad, bias=False),
                    norm_layer(ndf*nf), nn.LeakyReLU(0.2, True)]
        nf_prev = nf; nf = min(2**n_layers, 8)
        seq += [nn.Conv2d(ndf*nf_prev, ndf*nf, kw, 1, pad, bias=False),
                norm_layer(ndf*nf), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf*nf, 1, kw, 1, pad)]
        self.model = nn.Sequential(*seq)

    def forward(self, x): return self.model(x)
