import torch as t
import torch.nn as nn
from models import wideresnet
import models
from models import wideresnet_yopo
im_sz = 32
n_ch = 3


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, model='wrn', args=None):
        super(F, self).__init__()
        # default, wrn
        self.norm = norm
        if model == 'yopo':
            self.f = wideresnet_yopo.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        else:
            self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def feature(self, x):
        penult_z = self.f(x, feature=True)
        return penult_z

    def forward(self, x, y=None):
        penult_z = self.f(x, feature=True)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x, feature=True)
        output = self.class_output(penult_z).squeeze()
        return output


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, model='wrn', args=None):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes, model=model, args=args)

    def forward(self, x, y=None):
        logits = self.classify(x)

        if y is None:
            v = logits.logsumexp(1)
            # print("log sum exp", v)
            return v
        else:
            return t.gather(logits, 1, y[:, None])


def init_random(args, bs):
    im_sz = 32
    if args.dataset == 'tinyimagenet':
        im_sz = 64
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(args, device):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes, model=args.model)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer
