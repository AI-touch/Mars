import torch.nn as nn
import torch
import torch.nn.functional as F

import sys

sys.path.append('..')



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mars_saveloss_Net(nn.Module):
    def __init__(self, model_param):
        super(Mars_saveloss_Net, self).__init__()
        # The inputs to these layers will be passed through msaf before being passed into the layer

        if "video" in model_param:
            video_model = model_param["video"]["model"]
            # video model layers
            video_model = nn.Sequential(
                video_model.conv1,  # 0
                video_model.bn1,  # 1
                video_model.maxpool,  # 2
                video_model.layer1,  # 3
                video_model.layer2,  # 4
                video_model.layer3,  # 5
                video_model.layer4,  # 6
                video_model.avgpool,  # 7
                Flatten(),  # 8
                video_model.fc  # 9
            )
            self.video_model_blocks = video_model
            self.video_id = model_param["video"]["id"]
            # print("########## Video ##########")
            # for vb in self.video_model_blocks:
            #     print(vb)

        if "audio" in model_param:
            audio_model = model_param["audio"]["model"]
            # audio model layers
            audio_model = nn.Sequential(
                audio_model.conv1,  # 0
                nn.ReLU(inplace=True),  # 1
                audio_model.bn1,  # 2
                audio_model.conv2,  # 3
                nn.ReLU(inplace=True),  # 4
                audio_model.maxpool,  # 5
                audio_model.bn2,  # 6
                audio_model.dropout1,  # 7
                audio_model.conv3,  # 8
                nn.ReLU(inplace=True),  # 9
                audio_model.bn3,  # 10
                audio_model.flatten,  # 11
                audio_model.dropout2,  # 12
                audio_model.fc1  # 13
            )
            self.audio_model_blocks = audio_model
            self.audio_id = model_param["audio"]["id"]
            # print("########## Audio ##########")
            # for ab in self.audio_model_blocks:
            #     print(ab)

    def forward(self, x, y, global_step, para, l):
        if self.training == True:
            pred_v = self.video_model_blocks(x[self.video_id])
            pred_a = self.audio_model_blocks(x[self.audio_id])
            loss_v = F.cross_entropy(pred_v, y, reduction='none')
            log_pt = -loss_v
            pt = torch.exp(log_pt)
            weight_v = torch.exp(para[0] * (pt - para[1]))
            # weight = para[0] * torch.exp((pt - para[1]))

            floss_v = weight_v * loss_v
            floss_v = floss_v.mean()

            loss_a = F.cross_entropy(pred_a, y, reduction='none')
            log_pt = -loss_a
            pt = torch.exp(log_pt)
            weight_a = torch.exp(para[2] * (pt - para[3]))
            # weight = para[2] * torch.exp((pt - para[3]))
            # weight = pt ** 0.5
            floss_a = weight_a * loss_a
            floss_a = floss_a.mean()

            # loss_total = F.cross_entropy(pred_v + pred_a, y, reduction='mean')

            return pred_v + pred_a, floss_v + floss_a, loss_v.tolist(), loss_a.tolist(), weight_v.tolist(), weight_a.tolist()
            # if global_step < l:
            #     pred_v = self.video_model_blocks(x[self.video_id])
            #     pred_a = self.audio_model_blocks(x[self.audio_id])
            #     loss_v = F.cross_entropy(pred_v, y, reduction='none')
            #     loss_a = F.cross_entropy(pred_a, y, reduction='none')
            #
            #     return pred_v + pred_a, torch.mean(loss_v) + torch.mean(loss_a), loss_v.tolist(), loss_a.tolist()
            #
            # else:  #loss 相加，没有模型融合
            #     pred_v = self.video_model_blocks(x[self.video_id])
            #     pred_a = self.audio_model_blocks(x[self.audio_id])
            #     loss_v = F.cross_entropy(pred_v, y, reduction='none')
            #     log_pt = -loss_v
            #     pt = torch.exp(log_pt)
            #     weight = torch.exp(para[0]*(pt-para[1]))
            #     # weight = para[0] * torch.exp((pt - para[1]))
            #
            #     floss_v = weight * loss_v
            #     floss_v = floss_v.mean()
            #
            #     loss_a = F.cross_entropy(pred_a, y, reduction='none')
            #     log_pt = -loss_a
            #     pt = torch.exp(log_pt)
            #     weight = torch.exp(para[2]*(pt-para[3]))
            #     # weight = para[2] * torch.exp((pt - para[3]))
            #     # weight = pt ** 0.5
            #     floss_a = weight * loss_a
            #     floss_a = floss_a.mean()
            #
            #     # loss_total = F.cross_entropy(pred_v + pred_a, y, reduction='mean')
            #
            #     return pred_v + pred_a, floss_v + floss_a, loss_v.tolist(), loss_a.tolist()

        else:
            if global_step < 100:
                pred_v = self.video_model_blocks(x[self.video_id])
                pred_a = self.audio_model_blocks(x[self.audio_id])
                loss_v = F.cross_entropy(pred_v, y, reduction='mean')
                loss_a = F.cross_entropy(pred_a, y, reduction='mean')

                return pred_v , pred_a, loss_v + loss_a

            # else:
            #     pred_v = self.video_model_blocks(x[self.video_id])
            #     pred_a = self.audio_model_blocks(x[self.audio_id])
            #     prob_v = F.log_softmax(pred_v, dim=1)
            #     prob_a = F.log_softmax(pred_a, dim=1)
            #     pred = prob_v + prob_a
            #     loss = F.nll_loss(pred, y)
            #

            #     return pred, loss
