import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Dense_Corr(torch.nn.Module):
    def __init__(self, feat_model="resnet", devs=[0,1]):

        super(Dense_Corr, self).__init__()
        self.dev_ids = devs
        with torch.cuda.device(self.dev_ids[0]):
            self.activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    self.activation[name] = output.detach()
                return hook

            self.feat_model_st = feat_model.lower()

            if self.feat_model_st == "mobilenet":
                self.feat_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).cuda()
            elif self.feat_model_st == "resnet":
                self.feat_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).cuda()
                # for name, child in self.feat_model.layer3[22].named_children():
                #     print(name)
                self.feat_model.layer3[5].bn3.register_forward_hook(get_activation('ftl'))
            elif self.feat_model_st == "resnet34":
                self.feat_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True).cuda()
                self.feat_model.layer3[5].bn2.register_forward_hook(get_activation('ftl'))
            else:
                print("Setting for", self.feat_model_st, "not found. Using ResNet.")
                self.feat_model_st = "resnet"
                self.feat_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).cuda()
                self.feat_model.layer3[5].bn3.register_forward_hook(get_activation('ftl'))
            self.feat_model.eval()

        # self.preprocess = transforms.Compose([
        #     transforms.Resize([480, 640]),
        #     transforms.ToTensor()
        # ])
        # self.toPIL = transforms.ToPILImage(mode='RGB')

        with torch.cuda.device(self.dev_ids[1]):
            cur_dev = torch.cuda.current_device()
            self.dense = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 32, 3, padding=[1, 1]),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv2d(256, 256, 3, padding=[1, 1], stride=2),
                    # torch.nn.BatchNorm2d(256),
                    # torch.nn.Conv2d(256, 128, 3, padding=[1, 1]),
                    # torch.nn.BatchNorm2d(128),
                    # torch.nn.Conv2d(128, 64, 1, padding=[1, 1]),
                    # torch.nn.BatchNorm2d(64)
            ).cuda().to(cur_dev)

            self.SEblock = SELayer(15*20).cuda().to(cur_dev)

            self.convblock2 = torch.nn.Sequential(
                torch.nn.Conv2d(15*20+3+3, 128, 3, padding=[1, 1]),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 12, 3, padding=[1, 1], stride=2),
                torch.nn.BatchNorm2d(12),
                torch.nn.ReLU(inplace=True),
                # torch.nn.Conv2d(64, 12, 3, padding=[1, 1], stride=2),
                # torch.nn.BatchNorm2d(12)
            ).cuda().to(cur_dev)

            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(12*8*10, 256),
                torch.nn.Tanh(),
                # torch.nn.ReLU(inplace=True),
                # torch.nn.Linear(256, 256),
                # torch.nn.ReLU(inplace=True)
            ).cuda().to(cur_dev)

            self.n_reducer = torch.nn.Sequential(
                torch.nn.Conv2d(32, 10, 3, padding=[1, 1], stride=2),
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(inplace=True)
            ).cuda().to(cur_dev)

            self.n_encoder = torch.nn.Sequential(
                torch.nn.Linear(10*12*15, 256),
                torch.nn.Tanh(),
                # torch.nn.ReLU(inplace=True)
                # torch.nn.Linear(256, 256),
                # torch.nn.ReLU(inplace=True)
            ).cuda().to(cur_dev)

        # if self.feat_model_st == "resnet":
            # for name, child in self.feat_model.layer4[0].named_children():
            #     print(name)

    def forward(self, imgs):
        #imgs: shape:[2*E, 3, H, W], [:E] are left side of the pair, and [E:] are the respective right side of the pairs

        #Backbone: Feature Extractor
        with torch.cuda.device(self.dev_ids[0]):
            _ = self.feat_model(imgs.to(self.dev_ids[0]))
            node_feats = self.activation['ftl']

        with torch.cuda.device(self.dev_ids[1]):
            cur_dev = torch.cuda.current_device()
            self.dense.to(cur_dev)
            N = imgs.size(0)
            #Finding Correspondences
            opt = self.dense(node_feats.to(cur_dev))
            opt2 = F.interpolate(opt, [15, 20])
            vectors = opt2.view(N, 32, -1).permute(0, 2, 1)
            vectors = F.normalize(vectors, dim=2)
            h = N // 2
            v1 = vectors[:h].clone()
            v2 = vectors[h:].clone().permute(0, 2, 1)

            sim = torch.bmm(v1, v2)

            sim = F.softmax(sim, dim=2)

            self.SEblock.to(cur_dev)
            self.convblock2.to(cur_dev)
            self.encoder.to(cur_dev)

            att_sim = self.SEblock(sim.view(h, 15*20, 15, 20))
            # att_sim = sim.view(h, 15 * 20, 15, 20)
            im_rsz = F.interpolate(imgs, [15, 20])
            im_l = im_rsz[:h]
            im_r = im_rsz[h:]
            im = torch.cat((im_l, im_r), 1)
            im = im.to(cur_dev)
            cb2in = torch.cat((att_sim, im), 1)
            cb2out = self.convblock2(cb2in)
            e_enc = self.encoder(cb2out.view(h, -1))

            ## Node encoding
            n_enc = self.n_reducer(opt)
            n_enc = self.n_encoder(n_enc.view(N, -1))

        return sim, e_enc, n_enc