import argparse
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torchvision.models import resnet18, mobilenet_v2, vgg11
from utils import setup_logger
from dataset import get_loader
from autoencoder import AutoencoderHook
import torch.nn as nn

#train_ae(autoencoder)
def train(epochs, net, hook, optimizer, scheduler, train_loader, test_loader, gamma, logger):
    for e in range(epochs):
        net.eval()
        hook.train()
        for i, (images, labels) in enumerate(train_loader):
            inputs, labels = images.cuda(), labels.cuda()
            outputs = net(inputs)
            loss_classify = F.cross_entropy(outputs, labels)
            loss_reconstruction = F.smooth_l1_loss(hook.output, hook.input)
            loss = loss_reconstruction + gamma * loss_classify

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print(f'Train epoch {e}/{args.epochs}, Loss: {loss.item():.4f}')
        logger.info(f'Train epoch {e}/{args.epochs}, Loss: {loss.item():.4f}')
        test(net, hook, test_loader, logger)

def finetune(epochs, net, hook, optimizer_net, optimizer, train_loader, test_loader, logger):
    for e in range(epochs):
        net.train()
        hook.train()
        for i, (images, labels) in enumerate(train_loader):
            inputs, labels = images.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)

            optimizer_net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer_net.step()
            optimizer.step()
        print(f'Finetune epoch {e}/{epochs}, Loss: {loss.item():.4f}')
        logger.info(f'Finetune epoch {e}/{epochs}, Loss: {loss.item():.4f}')
        test(net, hook, test_loader, logger)

def test(net, hook, test_loader, logger):
    net.eval()   #hppo.py-eval()
    hook.eval()
    total_correct = 0
    losses = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            inputs, labels = images.cuda(), labels.cuda()
            outputs = net(inputs)
            losses += F.cross_entropy(outputs, labels).item()
            pred = outputs.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum().item()
    loss = round(losses / len(test_loader.dataset), 4)
    acc = round(total_correct / len(test_loader.dataset), 4)
    print(f'Test Avg. Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    logger.info(f'Test Avg. Loss: {loss:.4f}, Accuracy: {acc:.4f}')

# CUDA_VISIBLE_DEVICES=1 python train_ae_CTScan.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--ckp', type=str, default='ckp/')
    parser.add_argument('--log', type=str, default='log/')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenetv2', 'vgg11'])
    parser.add_argument('--point', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='balance the two part loss')
    parser.add_argument('--factor', type=int, default=24, help='shrink factor')  #32
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')   #vgg11:0.01  others：0.1
    parser.add_argument('--lr_f', type=float, default=1e-4,
                        help='finetune learning rate')
    parser.add_argument('--epochs', type=int, default=200,  #10
                        help='num of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')  #64
    parser.add_argument('--finetune', action='store_true', default=False) #False
    args = parser.parse_args()

    exp_name = f'{args.model}_point{args.point}_gamma{args.gamma}_factor{args.factor}' + \
        ('_finetune' if args.finetune else '')
    os.makedirs(os.path.join('CTScanresult', exp_name), exist_ok=True)
    logger = setup_logger(__name__, os.path.join('CTScanresult', exp_name))
    logger.info(args)

    if args.model == 'resnet18':
        net = resnet18(num_classes=4).cuda()
        modules = [net.layer1, net.layer2[1], net.layer3[1], net.layer4[1]]
    elif args.model == 'mobilenetv2':
        net = mobilenet_v2(num_classes=4).cuda()
        modules = [net.features[2], net.features[3],
                   net.features[5], net.features[8]]
    elif args.model == 'vgg11':
        net = vgg11(num_classes=4).cuda()
        modules = [net.features[3], net.features[6],
                   net.features[11], net.features[16]]
    else:
        raise NotImplementedError

    #+++++++

    #ckpt.pop('features.8.weight')
    # ckpt.pop('fc3.0.weight')
    #net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(os.path.join(
        'CTScanresult', f'CTScanmodel_train_{args.model}', 'CTScanckp.pt')), strict=False)

    #++++vgg11+++++
    # state_dict = torch.load(os.path.join('CTScanresult', f'CTScanmodel_train_{args.model}', 'CTScanckp.pt'))
    # model_dict = net.state_dict()
    # filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    # model_dict.update(filtered_dict)
    # net.load_state_dict(model_dict, strict=False)

    module = modules[args.point]
    hook = AutoencoderHook(net, module, args.factor,8)  #++++++qbit=8

    train_loader, test_loader = get_loader('CT_scan', args.batch_size)

    if not args.finetune:
        optimizer = optim.SGD(hook.parameters(), args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [7], 0.1)
        train(args.epochs, net, hook, optimizer, scheduler,
              train_loader, test_loader, args.gamma, logger)
    else:  #+++++++
        hook.load_state_dict(torch.load(
            args.ckp + exp_name[:-9] + '.pth')['ae_state'], strict=False) # strict=False
        optimizer = optim.SGD(hook.parameters(), args.lr_f,
                              momentum=0.9, weight_decay=5e-4)
        optimizer_net = optim.SGD(
            net.parameters(), args.lr_f, momentum=0.9, weight_decay=5e-4)
        finetune(5, net, hook, optimizer_net, optimizer,
                 train_loader, test_loader, logger)
    torch.save({'net_state': net.state_dict(), 'ae_state': hook.state_dict(
    )}, os.path.join('CTScanresult', exp_name, 'CTScanckp.pt'))
