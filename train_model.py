import argparse
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from utils import setup_logger
from dataset import get_loader

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--ckp', type=str, default='ckp/')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'vgg11', 'mobilenetv2'])
    parser.add_argument('--lr', type=float, default=0.01)   #resnet18/mobilenetv2 0.1  ;  vgg11  0.01
    parser.add_argument('--epochs', type=int, default=200)  #200
    parser.add_argument('--batch_size', type=int, default=64)  #64
    args = parser.parse_args()

    exp_name = f'CTScanmodel_train_{args.model}'
    os.makedirs(os.path.join('CTScanresult', exp_name), exist_ok=True)
    logger = setup_logger(__name__, os.path.join('CTScanresult', exp_name))

    if args.model == 'resnet18':
        net = torchvision.models.resnet18(num_classes=4).cuda()
    elif args.model == 'vgg11':
        net = torchvision.models.vgg11_bn(num_classes=4).cuda()
        #print(f'The current device：{torch.cuda.current_device()}')
        #CUDA_VISIBLE_DEVICES=4,5 python train_model.py
    elif args.model == 'mobilenetv2':
        net = torchvision.models.mobilenet_v2(num_classes=4).cuda()
    else:
        raise NotImplementedError

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], 0.1)
    #改++++++++
    train_loader, test_loader = get_loader('CT_scan', args.batch_size)

    #+++++++++++++
    train_losses = []
    test_losses = []
    test_accuracies = []

    for e in range(args.epochs):
        # train
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            inputs, labels = images.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.info(f'Epoch {e}/{args.epochs}, Loss: {loss.item():.4f}')

        # +++++++++++++
        # loss
        train_losses.append(loss.item())

        # test
        net.eval()
        total_correct = 0
        losses = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                inputs, labels = images.cuda(), labels.cuda()
                outputs = net(inputs)
                losses += F.cross_entropy(outputs, labels).item()
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)
                                         ).sum().item()
        loss = round(losses / len(test_loader.dataset), 4)
        acc = round(total_correct / len(test_loader.dataset), 4)
        logger.info(f'Test Avg. Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # +++++++++++++
        test_losses.append(loss)
        test_accuracies.append(acc)

        # save model
        torch.save(net.state_dict(), os.path.join('CTScanresult', exp_name, 'CTScanckp.pt'))

    # +++++++++++++pic
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('The accuracy of training the vgg11 classification model')
    plt.legend()
    plt.savefig('/home/zsq/works/MOCI/CTScanresult/CTScanmodel_train_vgg11/pics/CTScanacc_vgg11.png')
    plt.show()
    plt.clf()

    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('The loss of training and testing the vgg11 classification model')
    plt.legend()
    plt.savefig('/home/zsq/works/MOCI/CTScanresult/CTScanmodel_train_vgg11/pics/CTScanloss_vgg11.png')
    plt.show()
    plt.clf()



