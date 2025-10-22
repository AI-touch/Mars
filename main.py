
import glob
import math
import os
import datetime
import argparse
import random
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as transforms
from networks import *
from ravdess import RAVDESSDataset
from mars_utils import train, validation, visulize_loss, visualize_accuracy
from sklearn.metrics import confusion_matrix
from thop import profile
from thop import clever_format
from scipy.stats import pareto, powerlaw, zipf, gamma, expon

# setting seed
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# define model input
def get_X(device, sample):
    images = sample["images"].to(device)
    images = images.permute(0, 2, 1, 3, 4)  # swap to be (N, C, D, H, W)
    mfcc = sample["mfcc"].to(device)
    n = images[0].size(0)
    return [images, mfcc], n

def result_save(name, item, title):
    f = os.path.join(name, '{}.npy'.format(title))
    np.save(f, item)
    print('Saved result:', f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory',
                        default='RAVDESS/preprocessed')
    parser.add_argument('--k_fold', type=int, help='k for k fold cross validation', default=6)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--batch_size', type=int, help='batch size', default=2)
    parser.add_argument('--num_workers', type=int, help='num workers', default=4)
    parser.add_argument('--epochs', type=int, help='train epochs', default=30)
    parser.add_argument('--checkpointdir', type=str, help='directory to save/read weights',
                        default='checkpoints')
    parser.add_argument('--no_verbose', action='store_true', default=False, help='turn off verbose for training')
    parser.add_argument('--log_interval', type=int, help='interval for displaying training info if verbose', default=10)
    parser.add_argument('--no_save', action='store_true', default=False, help='set to not save model weights')
    parser.add_argument('--train', action='store_true', default=True, help='training')

    args = parser.parse_args()

    print("The configuration of this run is:")
    print(args, end='\n\n')
    project = 'marsloss_1'
    save_path = './result/' + project
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': True} \
        if use_cuda else {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}

    # define data transform
    train_transform = {
        "image_transform": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": None
    }

    val_transform = {
        "image_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": None
    }

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # top k categorical accuracy: k
    training_topk = (1,)
    val_topk = (1, 2, 4,)

    all_folder = sorted(list(glob.glob(os.path.join(args.datadir, "Actor*"))))
    s = int(len(all_folder) / args.k_fold)  # size of a fold
    top_scores = []
    pa_all = [[1.8, 1.8], [2.2, 2.2], [2.2, 2.2], [1.8, 2.4], [1.8, 1.8], [2, 2]]
    test_y = []
    test_pre_y = []
    test_pre_v = []
    test_pre_a = []
    for i in range(args.k_fold):
        print("Fold " + str(i + 1))
        resultdir = os.path.join(save_path, str(i + 1))
        os.makedirs(resultdir, exist_ok=True)

        # define dataset
        if args.train:
            train_fold = all_folder[:i * s] + all_folder[i * s + s:]
            training_set = RAVDESSDataset(train_fold, transform=train_transform)
            training_loader = data.DataLoader(training_set, **params)
            print("Train fold: ")
            print([os.path.basename(act) for act in train_fold])

            # record training process
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(args.checkpointdir, 'logs/fold{}_{}'.format(i + 1, current_time))
            # writer = SummaryWriter(log_dir=train_log_dir)

        val_fold = all_folder[i * s: i * s + s]
        val_set = RAVDESSDataset(val_fold, transform=val_transform)
        val_loader = data.DataLoader(val_set, **params)
        print("val fold: ")
        print([os.path.basename(act) for act in val_fold])

        # define model
        # video model
        model_param = {}
        video_model = resnet50(
            num_classes=8,
            shortcut_type='B',
            cardinality=32,
            sample_size=224,
            sample_duration=30
        )

        # input = torch.randn(1, 3, 30, 224, 224)
        # flops, params = profile(video_model, inputs=(input,))
        # flops, params = clever_format([flops, params], '%.3f')
        # print(flops, params)

        audio_model = MFCCNet()

        if args.train:  # load unimodal model weights
            video_model_path = os.path.join(args.checkpointdir,
                                            "resnext50/fold_{}_resnext50_best.pth".format(i + 1))
            video_model_checkpoint = torch.load(video_model_path) if use_cuda else \
                torch.load(video_model_path, map_location=torch.device('cpu'))
            video_model.load_state_dict(video_model_checkpoint)
            audio_model_path = os.path.join(args.checkpointdir,
                                            "mfccNet/fold_{}_mfccNet_best.pth".format(i + 1))
            audio_model_checkpoint = torch.load(audio_model_path) if use_cuda else \
                torch.load(audio_model_path, map_location=torch.device('cpu'))
            audio_model.load_state_dict(audio_model_checkpoint)

        model_param = {
            "video": {
                "model": video_model,
                "id": 0
            },
            "audio": {
                "model": audio_model,
                "id": 1
            }
        }
        multimodal_model = Mars_saveloss_Net(model_param)
        multimodal_model.to(device)

        # train / evaluate models
        if args.train:
            # Adam parameters
            num_parameters = multimodal_model.parameters()
            optimizer = torch.optim.Adam(num_parameters, lr=args.lr)
            # keep track of epoch test scores
            test = []
            best_acc_1 = 0
            los_v = []
            los_a = []
            acc_all = []
            acc_v = []
            acc_a = []
            d = 1
            l=15
            first_index = []
            found_d_below = False

            for epoch in range(args.epochs):
                if epoch == 0:
                    gama_v = mu_v = gama_a = mu_a = 0
                    para = [gama_v, mu_v, gama_a, mu_a]
                elif epoch < 10:
                    gama_v = mu_v = gama_a = mu_a = 0
                    para = [gama_v, mu_v, gama_a, mu_a]
                    data_v = los_v[epoch - 1]
                    data_a = los_a[epoch - 1]
                    ave_v = np.average(data_v)
                    ave_a = np.average(data_a)
                    d = abs(ave_v - ave_a)
                else:
                    data_v = los_v[epoch - 1]
                    data_a = los_a[epoch - 1]
                    ave_v = np.average(data_v)
                    ave_a = np.average(data_a)
                    d = abs(ave_v - ave_a)
                    if not found_d_below:
                        if d > 0.05:
                            gama_v = mu_v = gama_a = mu_a = 0
                            para = [gama_v, mu_v, gama_a, mu_a]

                        else:
                            gama_v = pa_all[i][0]
                            mu_v = 1
                            gama_a = pa_all[i][1]
                            mu_a = 1
                            para = [gama_v, mu_v, gama_a, mu_a]
                            found_d_below = True
                    else:
                        gama_v = pa_all[i][0]
                        mu_v = 1
                        gama_a = pa_all[i][1]
                        mu_a = 1
                        para = [gama_v, mu_v, gama_a, mu_a]


                print('两个模态损失差:', d)
                print(para)
                print('第几个epoch达到均衡：',first_index)
                # train, test model
                train_losses, train_scores, los_v_save, los_a_save,_,_ = train(get_X, args.log_interval, multimodal_model, device, training_loader,
                                                   optimizer, loss_func, training_topk, epoch, para,l)
                los_v.append(los_v_save)
                los_a.append(los_a_save)
                epoch_test_loss, epoch_test_score, v_score, a_score,_ = validation(get_X, multimodal_model, device, loss_func, val_loader,
                                                               val_topk,epoch=epoch)
                acc_all.append(epoch_test_score[0])
                acc_v.append(v_score[0])
                acc_a.append(a_score[0])
                if not args.no_save and epoch_test_score[0] > best_acc_1:
                    best_acc_1 = epoch_test_score[0]
                    torch.save(multimodal_model.state_dict(),
                               os.path.join(resultdir, 'fold_{}_save_mars8_best.pth'.format(i + 1)))
                    print("Epoch {} model saved!".format(epoch + 1))
                test.append(epoch_test_score)
                # writer.flush()
            test = np.array(test)
            for j, each_k in enumerate(val_topk):
                max_idx = np.argmax(test[:, j])
                print('Best top {} test score {:.2f}% at epoch {}'.format(each_k, test[:, j][max_idx], max_idx + 1))
            top_scores.append(test[:, 0].max())
            result_save(resultdir, los_v, 'los_v')
            result_save(resultdir, los_a, 'los_a')
            result_save(resultdir,acc_all,'acc')
            result_save(resultdir, acc_v, 'acc_v')
            result_save(resultdir, acc_a, 'acc_a')
            visulize_loss(los_v,los_a,'loss',resultdir)
            visualize_accuracy(acc_all,acc_v,acc_a,'acc',resultdir)

        else:  # load and evaluate model
            model_path = os.path.join(args.checkpointdir, 'fold_{}_save_focalloss_best.pth'.format(i + 1))
            checkpoint = torch.load(model_path) if use_cuda else torch.load(model_path,
                                                                            map_location=torch.device('cpu'))
            multimodal_model.load_state_dict(checkpoint)
            epoch_test_loss, epoch_test_score, v_score, a_score, pre = validation(get_X, multimodal_model, device, loss_func, val_loader,
                                                           val_topk,11)
            top_scores.append(epoch_test_score[0])
            test_y.extend(pre[0])
            test_pre_y.extend(pre[1])
            test_pre_v.extend(pre[2])
            test_pre_a.extend(pre[3])


        print("Scores for each fold: ")
        print(top_scores)
        print("Averaged score for {} fold: {:.2f}%".format(args.k_fold, sum(top_scores) / len(top_scores)))
    save_path = './result-fl/'
    cm = confusion_matrix(test_y, test_pre_y)
    print("Confusion matrix")
    print(cm)
    result_save(save_path, test_y, 'y_true')
    result_save(save_path, test_pre_y, 'test_pre_y')
    result_save(save_path, test_pre_v, 'test_pre_v')
    result_save(save_path, test_pre_a, 'test_pre_a')
