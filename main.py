import argparse
import time
import torch.cuda
import wandb
from common.config import Config
from dataset.create_data_loaders import get_data_loaders
from differential_privacy.get_private_model import get_private_model
from models.conv_net_with_centroids import ConvNetWithCentroids
from models.mlp_net import MlpNet
from models.conv_net import ConvNet
from train.utils import get_loss_and_opt, save_model, train_method

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single_train(batch_size, learning_rate, epochs=50, save_final_model=False):
    trainloader, testloader = get_data_loaders(dataset_name=Config.DATASET, batch_size=batch_size)
    print('train loader', len(trainloader))
    # net = MlpNet()  # ConvNet().to(device)
    net = ConvNetWithCentroids()
    net.to(device)
    criterion, optimizer = get_loss_and_opt(net, learning_rate=learning_rate)

    # priv_model, priv_optimizer, priv_loader = get_private_model(model=net, optimizer=optimizer, loader=trainloader,
    #                                                             batch_size=batch_size)

    priv_model, priv_optimizer, priv_loader = net, optimizer, trainloader

    train_method(trainloader=priv_loader, testloader=testloader, net=priv_model, criterion=criterion,
                 optimizer=priv_optimizer, epochs=epochs, device=device,
                 save_model_every=100)
    if save_final_model:
        filename_prefix = \
            f'{Config.SAVED_MODELS_DIR}' \
            f'model_{time.asctime()}' \
            f'_lr_{learning_rate}_batch_{batch_size}'
        save_model(model=priv_model, path=filename_prefix)


def run_single_train():
    GRAD_OR_SIGN = 'sgd'
    assert GRAD_OR_SIGN in ['sgd', 'sign', 'grad', 'sgd_dp', 'sgd_dp_sign', 'sgd_sign_dp']

    BATCH_SIZE = 256
    EPOCHS = 200
    LEARNING_RATE = 0.01

    wandb.init(project="emg_gp_moshe", entity="emg_diff_priv",
               name=f'SampleBatchDist lr {LEARNING_RATE} batch size {BATCH_SIZE}')
    wandb.config.update({'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE})

    single_train(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, save_final_model=True)


def run_sweep():
    # sweep_config = {
    #     'method': 'grid'
    # }
    # parameters_dict = {}
    #
    # sweep_config['parameters'] = parameters_dict
    # metric = {
    #     'name': 'val_acc',
    #     'goal': 'maximize'
    # }
    #
    # sweep_config['metric'] = metric
    #
    # # parameters_dict.update({
    # #     'epochs': {
    # #         'value': 50},
    # #     })
    #
    # parameters_dict.update({
    #     'learning_rate': {
    #         'values': [0.00001, 0.0001, 0.001]
    #     },
    #     'batch_size': {
    #         'values': [16, 32, 64]
    #     }
    # })
    #
    # sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    #
    # wandb.agent(sweep_id, sweep_train)
    pass


def main(args):
    if args.single_or_sweep == 'single':
        run_single_train()
    else:
        run_sweep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--single_or_sweep", type=str, choices=['single', 'sweep'], default='single')

    args = parser.parse_args()

    main(args)
