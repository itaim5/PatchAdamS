import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import torch.multiprocessing as tmp
from models.models import Normalize
from argparse import ArgumentParser
from models.models import MnistNet
from models.resnet import ResNet50
from config import *
from func import gen_adv
from flow import train, test
from vis import *
from instructions import PatchInitCfg


if __name__ == '__main__':
    parser = ArgumentParser(description='Adversarial Programs')
    parser.add_argument('-d', '--dataset', help='Dataset', required=True)
    parser.add_argument('-g', '--gpus', help='Allowed GPUs', required=True)
    parser.add_argument('-p', '--parallel', help='number of concurrent processes per GPU', required=True)
    parser.add_argument('-s', '--num_samples', help='number of samples to attack', required=True)
    parser.add_argument('-a', '--attacks_per_sample', help='number of attacks per sample', required=True)
    parser.add_argument('-i', '--instructions', help='number of maximal instructions (patch sequences)', required=True)
    parser.add_argument('-c', '--candidates', help='number of candidates', required=True)
    args = vars(parser.parse_args())
    print(args)

    # initialize GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpus']

    torch.cuda.empty_cache()
    print("running on device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # parameters
    prms = Params(args['dataset'], norm_power=2, batch_size=(32 if torch.cuda.is_available() else 1),
                  parallel=int(args['parallel']))

    kwargs = {'num_workers': 0, 'pin_memory': True, 'batch_size': prms.batch_size} if torch.cuda.is_available() else {}

    # organize datasets
    if prms.dataset == "mnist":
        train_data = datasets.MNIST('../data/mnist/', train=True, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
        test_data = datasets.MNIST('../data/mnist/', train=False, transform=transforms.Compose([transforms.ToTensor()]))
        prms.model = MnistNet()
    elif prms.dataset == "fashion-mnist":
        train_data = datasets.FashionMNIST('../data/fashion-mnist/', train=True, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
        test_data = datasets.FashionMNIST('../data/fashion-mnist/', train=False,
                                          transform=transforms.Compose([transforms.ToTensor()]))
        prms.model = MnistNet()
    elif prms.dataset == "cifar10":
        train_data = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))

        test_data = datasets.CIFAR10('../data/cifar10', train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
        prms.model = ResNet50()
    # to use ImageNet - please manually download ILSVRC files to "../data/imagenet"
    elif prms.dataset == "imagenet":
        test_data = datasets.ImageNet('../data/imagenet', split='val',
                                                          transform=transforms.Compose([
                                                              transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()]))

        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        prms.model = nn.Sequential(
            norm_layer,
            models.resnext50_32x4d(pretrained=True)
        )

    # train or load pre-trained model
    if prms.dataset != "imagenet":
        model_file = os.path.join("models", prms.dataset + "_model.pt")
        if os.path.isfile(model_file):
            prms.model.load_state_dict(torch.load(model_file, map_location='cpu'))
        else:
            train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)
            print("training model...")
            tmp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if prms.dataset == "mnist" or prms.dataset == "fashion-mnist":
                optimizer = optim.Adam(prms.model.parameters(), lr=0.0001)
                scheduler = None
            elif prms.dataset == "cifar10":
                optimizer = optim.SGD(prms.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 250], gamma=0.1)
                # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30)
            for epoch in range(1, prms.num_train_epochs + 1):
                train_acc, train_loss, epoch_time = train(prms, tmp_device, train_loader, optimizer, epoch)
                test_acc, test_loss = test(prms, tmp_device, test_loader)
                print('Epoch Time [s]:', epoch_time)
                if scheduler is not None:
                    scheduler.step()
                    # scheduler.step(test_loss)
            torch.save(prms.model.state_dict(), model_file)
            prms.model = prms.model.to('cpu')

    prms.model.eval()

    # # model accuracy check
    # test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)
    # prms.model = prms.model.to('cuda:0')
    # start = time.time()
    # test_acc, test_loss = test(prms, 'cuda:0', test_loader)
    # end = time.time()
    # prms.model = prms.model.to('cpu')
    # print("Inference took %d seconds" % (end-start))

    # divide the test set into class-specific sets
    # also, separate a small val set for instruction evaluation
    kwargs['batch_size'] = 1
    test_class_loaders, val_class_sets = prepare_class_sets(test_data, prms, kwargs)

    #### set runtime parameters ####
    num_samples = int(args['num_samples'])
    attacks_per_sample = int(args['attacks_per_sample'])
    total_attacks = num_samples * attacks_per_sample

    # job partitioning on cuda devices
    max_jobs_per_gpu = int(args['parallel'])
    num_gpus = torch.cuda.device_count()
    total_parallel_jobs = num_gpus * max_jobs_per_gpu
    kwargs['batch_size'] = num_samples

    devices = []
    if torch.cuda.is_available():
        print('Using', num_gpus, "GPUs")
        for i in range(num_gpus):
            devices.append(torch.device('cuda:' + str(i)))

        prll_jobs_vec = [total_parallel_jobs] * (total_attacks // total_parallel_jobs)
        if total_attacks % total_parallel_jobs:
            prll_jobs_vec.append(total_attacks % total_parallel_jobs)
    else:
        print('Using CPU')
        devices = ['cpu']
        prll_jobs_vec = [1] * num_samples * attacks_per_sample

    #### create a subset from the test set (3 samples per class) ####
    if prms.dataset == "imagenet":
        classes = [76, 231, 292, 315, 471, 562, 817, 857, 928, 979]
        indices = [3800, 3801, 3802, 11551, 11552, 11553, 14601, 14602, 14603, 15750, 15751, 15752, 23550, 23551, 23552,
                28100, 28101, 28102, 40850, 40852, 40853, 42851, 42852, 42853, 46401, 46404, 46405, 48951, 48953, 48955]
    else:
        classes = [x for x in range(prms.num_classes)]
        indices, _ = get_n_good_samples_per_class(test_data, 3, prms, classes)
    assert len(indices) == num_samples
    subset_data = torch.utils.data.Subset(test_data, indices)
    subset_loader = torch.utils.data.DataLoader(subset_data, shuffle=False, **kwargs)
    #################################################################

    root_dir = os.path.join("results_" + prms.dataset, "joint_optimization")
    cfgs = []
    for _, dirs, _ in os.walk(root_dir):
        for subdir in dirs:
            cfg = Config(subdir)
            cfgs.append(cfg)
        break

    try:
        num_instructions = int(args['instructions'])
        num_candidates = int(args['candidates'])

        if prms.dataset == "imagenet":
            cntous_lr = 0.05
            dscrt_lr = 2.5
            score_norm_reg = 0.05
            b1 = 0.1
            b2 = 0.3
            cb1 = 0.3
            cb2 = 0.3
            stride = 10
            lpt = 0.005
            lpp = 200
            optim_itr = 400
            stds = [0.3, 0.3]
            size_range = [15, 30]
        else:
            cntous_lr = 0.01
            dscrt_lr = 1
            score_norm_reg = 1
            b1 = 0.05
            b2 = 0.1
            cb1 = 0.3
            cb2 = 0.3
            stride = 3
            lpt = 0.005
            lpp = 25
            optim_itr = 300
            stds = [0.3, 0.3]
            size_range = [5,15]

            s_bounds = None
            e_bounds = None

            patch_init_cfg = PatchInitCfg(std_scale=stds[0],
                                          std_shift=stds[1],
                                          min_size=size_range[0],
                                          max_size=size_range[1])
            s_bounds = s_bounds if s_bounds is not None else [0,0,0]
            e_bounds = e_bounds if e_bounds is not None else prms.image_shape
            for s, e in zip(s_bounds[1:], e_bounds[1:]):
                assert e - s >= patch_init_cfg.min_size
                assert e - s >= patch_init_cfg.max_size

            norm_sum = 0.
            num_attacks = 0
            avg_norm = 0.
            print("############## %d instructions, clr = %.2f, dlr = %.2f, optitr = %d, snr = %.1f, "
                  "candidates = %d, lpt = %.4f, lpp = %d, init_size_range = [%d, %d], init_stds = [%.2f, %.2f] "
                  "##############" % (num_instructions, cntous_lr, dscrt_lr, optim_itr, score_norm_reg,
                                      num_candidates, lpt, lpp, size_range[0], size_range[1], stds[0], stds[1]))
            dir_name = "num_inst" + str(num_instructions) + "_clr" + str(cntous_lr) + "_dlr" + \
                       str(dscrt_lr) + "_optitr" + str(optim_itr) + "_snr" + "" + str(score_norm_reg) + "_cnd" + "" \
                       + str(num_candidates) + "_str" + str(stride) + "_lpt" + str(lpt) + "_lpp" + str(lpp) \
                       + "_stds" + str(stds) + "_sizes" + str(size_range) + "_s_bounds" + str(s_bounds) + \
                       "_e_bounds" + str(e_bounds)

            ###################################
            out_dir = os.path.join(root_dir, dir_name)
            if os.path.exists(out_dir):
                print("out dir exist")

            itr = iter(subset_loader)
            x_batch, true_lbls = next(itr)

            print(prll_jobs_vec)
            glob_attack_ctr = 0
            for prll_jobs in prll_jobs_vec:
                ctx = tmp.get_context('spawn')
                queue = tmp.Manager().Queue()
                processes = []
                for ii in range(prll_jobs):
                    # set device
                    gpu_idx = ii % num_gpus
                    device = devices[gpu_idx]
                    # get image from batch
                    image_idx = glob_attack_ctr // attacks_per_sample
                    x_image, true_label = x_batch[image_idx:image_idx + 1].to(device), true_lbls[image_idx].to(device)
                    # choose target label
                    trgt_lbls_list = [x for x in classes if x != true_label.item()]
                    assert len(trgt_lbls_list) >= attacks_per_sample
                    if len(trgt_lbls_list) > attacks_per_sample:
                        trgt_lbls_list = trgt_lbls_list[:attacks_per_sample]
                    trgt_lbls = torch.tensor(trgt_lbls_list, dtype=torch.int64)
                    assert trgt_lbls.shape[0] == attacks_per_sample
                    trgt_idx = glob_attack_ctr % attacks_per_sample
                    target_label = trgt_lbls[trgt_idx].to(device)

                    print("ii=%d, GPU %d , image %d , target_idx %d" % (ii, gpu_idx, image_idx, trgt_idx))
                    processes.append(ctx.Process(target=gen_adv, args=(x_image, true_label, target_label,
                                                                       num_instructions, optim_itr, score_norm_reg,
                                                                       num_candidates, lpt, lpp, cntous_lr,
                                                                       dscrt_lr, stride, (b1,b2), (cb1,cb2),
                                                                       patch_init_cfg, s_bounds, e_bounds, out_dir,
                                                                       device, prms, queue)))
                    glob_attack_ctr += 1

                for proc in processes:
                    proc.start()
                for proc in processes:
                    proc.join()
                while not queue.empty():
                    num_attacks += 1
                    norm_sum += queue.get()
                avg_norm = norm_sum / num_attacks if num_attacks else -1
                print("intermediate:  avg_norm = %f (num_attacks = %d)" % (avg_norm, num_attacks))

            avg_norm = norm_sum / num_attacks if num_attacks else -1
            print("finished:  avg_norm = %f (num_attacks = %d)" % (avg_norm, num_attacks))
            try:
                os.rename(out_dir, out_dir + "_num_attacks" + str(num_attacks) + "_norm" + str(round(avg_norm, 2)))
                out_dir = out_dir + "_num_attacks" + str(num_attacks) + "_norm" + str(round(avg_norm, 2))
            except:
                print("identical run, same norm")
            # calc_gen_rate(out_dir, test_class_loaders, prms, device)

    except Exception as ex:
        while not queue.empty():
            num_attacks += 1
            norm_sum += queue.get()
        avg_norm = norm_sum / num_attacks if num_attacks else -1
        print("finished (by exception):  avg_norm = %f (num_attacks = %d)" % (avg_norm, num_attacks))

        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
