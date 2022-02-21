import torch
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from utils import hls_to_rgb
from matplotlib.patches import Patch as mpPatch
import func
from utils import move_program_to_device


def imshow_wrp(image, dataset):
    if dataset == "mnist" or dataset == "fashion-mnist":
        plt.imshow(image[0, 0].detach().cpu(), "gray", vmin=0, vmax=1, interpolation='nearest')
    elif dataset == "cifar10" or dataset == "imagenet":
        plt.imshow(image[0].permute([1, 2, 0]).detach().cpu(), vmin=0, vmax=1, interpolation='nearest')
    else:
        print("unsupported dataset...")


def plot_patches(instructions, verbose=False):
    if instructions[0].scale.shape[0] == 3:
        num_channels = 3
    else:
        num_channels = 1

    patch_masks = [inst.patch.patch_mask.detach() for inst in instructions]
    num_patches = len(patch_masks)
    image = torch.zeros([3, patch_masks[0].shape[1], patch_masks[0].shape[2]])
    legend_elements = []
    for i in range(num_patches):
        if patch_masks[i].shape[0] == 1:
            patch_mask = patch_masks[i].repeat(3,1,1).cpu()
        else:
            patch_mask = patch_masks[i].cpu()
        rgb_color = hls_to_rgb(i/num_patches, 0.5, 0.9)
        patch_rgb = torch.tensor(rgb_color).unsqueeze(1).unsqueeze(2)
        image += patch_mask * patch_rgb
        if verbose:
            if num_channels == 1:
                inst_str = 'I(' + str(round(instructions[i].scale.item(), 2)) + ", " + \
                           str(round(instructions[i].shift.item(), 2)) + ")"
            else:
                inst_str = 'I(' + str(tuple([round(x.item(), 2) for x in instructions[i].scale[:,0,0]])) + ",\n" + \
                           str(tuple([round(x.item(), 2) for x in instructions[i].shift[:,0,0]])) + ")"
            legend_elements.append(mpPatch(facecolor=rgb_color, label=inst_str))
        else:
            legend_elements.append(mpPatch(facecolor=rgb_color, label='Patch%d' % i))

    image = torch.clamp(image, 0, 1)
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    if verbose:
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize='medium', prop={'size': 10, 'weight':'bold'}, borderaxespad=0.)
    else:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
    # plt.show()


def gen_iterative_exp_plot():
    num_cnd_vec = [3,5]
    num_inst_vec = [3,5]
    cnd_inst_str_vec = ["(3,3)", "(3,5)", "(5,3)", "(5,5)"]
    l2_norms = {x: ([None] * len(num_cnd_vec)*len(num_inst_vec)) for x in [1, 0]}
    success_rates = {x: ([None] * len(num_cnd_vec)*len(num_inst_vec)) for x in [1, 0]}
    attack_times = {x: ([None] * len(num_cnd_vec)*len(num_inst_vec)) for x in [1, 0]}

    dataset_rootdir = os.path.join("results_mnist", "joint_optimization", "iterative_new_exp_no_smart_init")
    for cfg_dir in os.listdir(dataset_rootdir):
        num_inst = re.match(".*num_inst(\d+)", cfg_dir)
        num_cnd = re.match(".*cnd(\d+)", cfg_dir)
        itr = re.match(".*itrInst(\d)", cfg_dir)
        if num_inst is not None and num_cnd is not None and itr is not None:
            num_instructions = int(num_inst.group(1))
            num_candidates = int(num_cnd.group(1))
            iterative = int(itr.group(1))
            global_stats = torch.load(os.path.join(dataset_rootdir, cfg_dir, "global_stats.pt"))

            l2_norms[iterative][num_cnd_vec.index(num_candidates) + 2 * num_inst_vec.index(num_instructions)] = global_stats['l2_norm']
            success_rates[iterative][num_cnd_vec.index(num_candidates) + 2 * num_inst_vec.index(num_instructions)] = global_stats['success_rate']
            attack_times[iterative][num_cnd_vec.index(num_candidates) + 2 * num_inst_vec.index(num_instructions)] = global_stats['attack_time']

    font = {'size': 20}
    plt.rc('font', **font)

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': l2_norms[1] , 'w/o synthesizer': l2_norms[0]}, index=cnd_inst_str_vec)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    ax1.set_xlabel("(# patches, # candidates)", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(6, 4.5)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "itr_per_inst_per_cnd_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': success_rates[1] , 'w/o synthesizer': success_rates[0]}, index=cnd_inst_str_vec)
    ax2 = df.plot.bar(rot=0, legend=False)
    ax2.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    ax2.set_xlabel("(# patches, # candidates)", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(6, 4.5)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "itr_per_inst_per_cnd_success_rates.png"))

    tmp = plt.figure()
    df = pd.DataFrame({'PatchAdamS': attack_times[1] , 'w/o synthesizer': attack_times[0]}, index=cnd_inst_str_vec)
    ax3 = df.plot.bar(rot=0)
    ax3.set_ylabel("Execution Time [m]", fontdict=dict(weight='bold'))
    ax3.set_xlabel("(# patches, # candidates)", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(6, 4.5)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "itr_per_inst_per_cnd_attack_times.png"))

    size = tmp.get_size_inches()
    print(size)


def gen_smart_init_plot():
    num_cnd_vec = [1,3,5,7]
    num_inst_vec = [1,3,5,7]
    cnd_inst_str_vec = ["(1,1)","(1,3)","(1,5)","(5,7)","(3,1)","(3,3)","(3,5)","(3,7)","(5,1)","(5,3)","(5,5)","(5,7)","(7,1)","(7,3)","(7,5)","(7,7)"]
    success_rates = {x: ([None] * len(num_cnd_vec)*len(num_inst_vec)) for x in [1, 0]}

    d = {True: os.path.join("results_mnist", "joint_optimization", "table_exp"),
         False: os.path.join("results_mnist", "joint_optimization", "wo_smart_init")}
    for bin, path in d.items():
        for cfg_dir in os.listdir(path):
            num_inst = re.match(".*num_inst(\d+)", cfg_dir)
            num_cnd = re.match(".*cnd(\d+)", cfg_dir)
            # num_att = re.match(".*num_attacks(\d+)", cfg_dir)
            if num_inst is not None and num_cnd is not None:
                num_instructions = int(num_inst.group(1))
                if num_instructions == 10:
                    continue
                num_candidates = int(num_cnd.group(1))
                # num_attacks = int(num_att.group(1))
                # sr = num_attacks / 270.
                global_stats = torch.load(os.path.join(path, cfg_dir, "global_stats.pt"))

                # if abs(sr * 100 - global_stats['success_rate']) > 0.05:
                #     print(bin, num_instructions, num_candidates)
                #     print(sr * 100, global_stats['success_rate'])
                success_rates[bin][num_cnd_vec.index(num_candidates) + 4 * num_inst_vec.index(num_instructions)] = global_stats['success_rate']

    font = {'size': 20}
    plt.rc('font', **font)

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': success_rates[True] , 'w/o PatchInit': success_rates[False]}, index=cnd_inst_str_vec)
    ax2 = df.plot.bar(rot=0)
    ax2.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    ax2.set_xlabel("(# patches, # candidates)", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(15, 4.5)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "smart_init_per_inst_per_cnd_success_rates.png"))

    # size = tmp.get_size_inches()
    # print(size)


def gen_per_inst_per_dataset_bar_plot():
    num_cnd_for_per_inst = 7
    num_inst_vec = [1,3,5,7,10]
    num_inst_for_per_cnd = 5
    num_cnd_vec = [1,3,5,7,10]
    datasets = ["mnist", "fashion-mnist", "cifar10"]

    all_datasets = ["mnist", "fashion-mnist", "cifar10", "imagenet"]
    imagenet_inst = [5,7,10]
    imagenet_cnd = [7,10]
    imagenet_l0_norms = {}
    imagenet_l2_norms = {}
    imagenet_success_rates = {}
    imagenet_attack_times = {}

    l0_norms = {x: ([None] * len(datasets)) for x in num_inst_vec}
    l0_norms['C&W'] = [None] * len(datasets)
    l2_norms = {x: ([None] * len(datasets)) for x in num_inst_vec}
    l2_norms['C&W'] = [None] * len(datasets)
    success_rates = {x: ([None] * len(datasets)) for x in num_inst_vec}
    success_rates['C&W'] = [None] * len(datasets)
    attack_times = {x: ([None] * len(datasets)) for x in num_inst_vec}
    attack_times['C&W'] = [None] * len(datasets)
    gen_rates = {x: ([None] * len(datasets)) for x in num_inst_vec}
    gen_rates['C&W'] = [None] * len(datasets)
    gen_l2_norms = {x: ([None] * len(datasets)) for x in num_inst_vec}
    gen_l2_norms['C&W'] = [None] * len(datasets)

    mnist_l2_norms_per_cnd = [None] * len(num_cnd_vec)
    mnist_l0_norms_per_cnd = [None] * len(num_cnd_vec)
    mnist_success_rates_per_cnd = [None] * len(num_cnd_vec)
    mnist_attack_times_per_cnd = [None] * len(num_cnd_vec)
    mnist_gen_rates_per_cnd = [None] * len(num_cnd_vec)
    mnist_gen_l2_norms_per_cnd = [None] * len(num_cnd_vec)
    fmnist_l2_norms_per_cnd = [None] * len(num_cnd_vec)
    fmnist_l0_norms_per_cnd = [None] * len(num_cnd_vec)
    fmnist_success_rates_per_cnd = [None] * len(num_cnd_vec)
    fmnist_attack_times_per_cnd = [None] * len(num_cnd_vec)
    fmnist_gen_rates_per_cnd = [None] * len(num_cnd_vec)
    fmnist_gen_l2_norms_per_cnd = [None] * len(num_cnd_vec)
    cifar10_l2_norms_per_cnd = [None] * len(num_cnd_vec)
    cifar10_l0_norms_per_cnd = [None] * len(num_cnd_vec)
    cifar10_success_rates_per_cnd = [None] * len(num_cnd_vec)
    cifar10_attack_times_per_cnd = [None] * len(num_cnd_vec)
    cifar10_gen_rates_per_cnd = [None] * len(num_cnd_vec)
    cifar10_gen_l2_norms_per_cnd = [None] * len(num_cnd_vec)

    # baseline 3 datasets
    for i, dataset in enumerate(all_datasets):
        dataset_rootdir = os.path.join("results_" + dataset, "joint_optimization", "table_exp")
        for cfg_dir in os.listdir(dataset_rootdir):
            num_inst = re.match(".*num_inst(\d+)", cfg_dir)
            num_cnd = re.match(".*cnd(\d+)", cfg_dir)
            if num_inst is not None and num_cnd is not None:
                num_instructions = int(num_inst.group(1))
                num_candidates = int(num_cnd.group(1))
                global_stats = torch.load(os.path.join(dataset_rootdir, cfg_dir, "global_stats.pt"))
                gen_stats = torch.load(os.path.join(dataset_rootdir, cfg_dir, "gen_stats.pt"))
                if dataset == "imagenet":
                    imagenet_l0_norms[(num_instructions,num_candidates)] = global_stats['l0_norm']
                    imagenet_l2_norms[(num_instructions,num_candidates)] = global_stats['l2_norm']
                    imagenet_success_rates[(num_instructions,num_candidates)] = global_stats['success_rate']
                    imagenet_attack_times[(num_instructions,num_candidates)] = global_stats['attack_time']
                else:
                    if num_candidates == num_cnd_for_per_inst:
                        l0_norms[num_instructions][i] = global_stats['l0_norm']
                        l2_norms[num_instructions][i] = global_stats['l2_norm']
                        success_rates[num_instructions][i] = global_stats['success_rate']
                        attack_times[num_instructions][i] = global_stats['attack_time']
                        gen_rates[num_instructions][i] = gen_stats['prog_ave_gen_suc_rate_t']
                        gen_l2_norms[num_instructions][i] = gen_stats['prog_ave_gen_norm_t']
                        if l2_norms['C&W'][i] is None:
                            l0_norms['C&W'][i] = global_stats['cw_l0_norm']
                            l2_norms['C&W'][i] = global_stats['cw_l2_norm']
                            success_rates['C&W'][i] = global_stats['cw_success_rate']
                            attack_times['C&W'][i] = global_stats['cw_attack_time']
                            gen_rates['C&W'][i] = gen_stats['cw_ave_gen_suc_rate_t']
                            gen_l2_norms['C&W'][i] = gen_stats['cw_ave_gen_norm_t']
                    if num_instructions == num_inst_for_per_cnd:
                        cnd_index = num_cnd_vec.index(num_candidates)
                        if dataset == "mnist":
                            mnist_l0_norms_per_cnd[cnd_index] = global_stats['l0_norm']
                            mnist_l2_norms_per_cnd[cnd_index] = global_stats['l2_norm']
                            mnist_success_rates_per_cnd[cnd_index] = global_stats['success_rate']
                            mnist_attack_times_per_cnd[cnd_index] = global_stats['attack_time']
                            mnist_gen_rates_per_cnd[cnd_index] = gen_stats['prog_ave_gen_suc_rate_t']
                            mnist_gen_l2_norms_per_cnd[cnd_index] = gen_stats['prog_ave_gen_norm_t']
                        elif dataset == "fashion-mnist":
                            fmnist_l0_norms_per_cnd[cnd_index] = global_stats['l0_norm']
                            fmnist_l2_norms_per_cnd[cnd_index] = global_stats['l2_norm']
                            fmnist_success_rates_per_cnd[cnd_index] = global_stats['success_rate']
                            fmnist_attack_times_per_cnd[cnd_index] = global_stats['attack_time']
                            fmnist_gen_rates_per_cnd[cnd_index] = gen_stats['prog_ave_gen_suc_rate_t']
                            fmnist_gen_l2_norms_per_cnd[cnd_index] = gen_stats['prog_ave_gen_norm_t']
                        elif dataset == "cifar10":
                            cifar10_l0_norms_per_cnd[cnd_index] = global_stats['l0_norm']
                            cifar10_l2_norms_per_cnd[cnd_index] = global_stats['l2_norm']
                            cifar10_success_rates_per_cnd[cnd_index] = global_stats['success_rate']
                            cifar10_attack_times_per_cnd[cnd_index] = global_stats['attack_time']
                            cifar10_gen_rates_per_cnd[cnd_index] = gen_stats['prog_ave_gen_suc_rate_t']
                            cifar10_gen_l2_norms_per_cnd[cnd_index] = gen_stats['prog_ave_gen_norm_t']

    datasets_cap = ["MNIST", "Fashion-MNIST", "CIFAR-10"]

    font = {'size': 20}
    plt.rc('font', **font)

    ##### no discrete optimization exp #####
    dexp_l0_norms = {x: ([None] * len(datasets)) for x in [True, False]}
    dexp_l2_norms = {x: ([None] * len(datasets)) for x in [True, False]}
    dexp_success_rates = {x: ([None] * len(datasets)) for x in [True, False]}

    # MNIST
    dexp_l2_norms[True][0] = 2.58
    dexp_l2_norms[False][0] = 3.28
    dexp_l0_norms[True][0] = 157.6
    dexp_l0_norms[False][0] = 228.3
    dexp_success_rates[True][0] = 1
    dexp_success_rates[False][0] = 0.97
    # Fashion-MNIST
    dexp_l2_norms[True][1] = 1.3
    dexp_l2_norms[False][1] = 2.11
    dexp_l0_norms[True][1] = 191.5
    dexp_l0_norms[False][1] = 290.7
    dexp_success_rates[True][1] = 1
    dexp_success_rates[False][1] = 1
    # CIFAR-10
    dexp_l2_norms[True][2] = 1.26
    dexp_l2_norms[False][2] = 1.87
    dexp_l0_norms[True][2] = 332.6
    dexp_l0_norms[False][2] = 1101.2
    dexp_success_rates[True][2] = 0.97
    dexp_success_rates[False][2] = 0.9

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': dexp_success_rates[True], 'fixed patches': dexp_success_rates[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "dscrt_opt_exp_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': dexp_l2_norms[True], 'fixed patches': dexp_l2_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0)
    ax1.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "dscrt_opt_exp_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': dexp_l0_norms[True], 'fixed patches': dexp_l0_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("L0 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "dscrt_opt_exp_l0_norms.png"))

    ##### no adaptive optimization exp #####
    adpt_l0_norms = {x: ([None] * len(datasets)) for x in [True, False]}
    adpt_l2_norms = {x: ([None] * len(datasets)) for x in [True, False]}
    adpt_success_rates = {x: ([None] * len(datasets)) for x in [True, False]}

    # MNIST
    adpt_l2_norms[True][0] = 2.58
    adpt_l2_norms[False][0] = 2.67  # 2.73
    adpt_l0_norms[True][0] = 157.6
    adpt_l0_norms[False][0] = 200  # 208.8
    adpt_success_rates[True][0] = 1
    adpt_success_rates[False][0] = 0.98
    # Fashion-MNIST
    adpt_l2_norms[True][1] = 1.3
    adpt_l2_norms[False][1] = 1.46  # 1.43
    adpt_l0_norms[True][1] = 191.5
    adpt_l0_norms[False][1] = 278  # 277.9
    adpt_success_rates[True][1] = 1
    adpt_success_rates[False][1] = 1
    # CIFAR-10
    adpt_l2_norms[True][2] = 1.26
    adpt_l2_norms[False][2] = 1.46  # 1.37
    adpt_l0_norms[True][2] = 332.6
    adpt_l0_norms[False][2] = 737.5  # 807.8
    adpt_success_rates[True][2] = 0.97
    adpt_success_rates[False][2] = 0.96  # 0.94

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': adpt_success_rates[True], 'w/o adaptive opt.': adpt_success_rates[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "adpt_opt_exp_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': adpt_l2_norms[True], 'w/o adaptive opt.': adpt_l2_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0)
    ax1.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "adpt_opt_exp_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': adpt_l0_norms[True], 'w/o adaptive opt.': adpt_l0_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("L0 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "adpt_opt_exp_l0_norms.png"))

    ##### no hill climbing exp #####
    hc_l0_norms = {x: ([None] * len(datasets)) for x in [True, False]}
    hc_l2_norms = {x: ([None] * len(datasets)) for x in [True, False]}
    hc_success_rates = {x: ([None] * len(datasets)) for x in [True, False]}

    # MNIST
    hc_l2_norms[True][0] = 2.58
    hc_l2_norms[False][0] = 2.73
    hc_l0_norms[True][0] = 157.6
    hc_l0_norms[False][0] = 119.05
    hc_success_rates[True][0] = 1
    hc_success_rates[False][0] = 0.996
    # Fashion-MNIST
    hc_l2_norms[True][1] = 1.3
    hc_l2_norms[False][1] = 1.57
    hc_l0_norms[True][1] = 191.5
    hc_l0_norms[False][1] = 155.53
    hc_success_rates[True][1] = 1
    hc_success_rates[False][1] = 1
    # CIFAR-10
    hc_l2_norms[True][2] = 1.26
    hc_l2_norms[False][2] = 1.46
    hc_l0_norms[True][2] = 332.6
    hc_l0_norms[False][2] = 221.05
    hc_success_rates[True][2] = 0.97
    hc_success_rates[False][2] = 0.956

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': hc_success_rates[True], 'w/o HC': hc_success_rates[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "hc_exp_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': hc_l2_norms[True], 'w/o HC': hc_l2_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0)
    ax1.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "hc_exp_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': hc_l0_norms[True], 'w/o HC': hc_l0_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("L0 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "hc_exp_l0_norms.png"))

    ########## unified #############

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': dexp_success_rates[True], 'w/o HC': hc_success_rates[False], 'w/o Adam': adpt_success_rates[False], 'w/o optimization': dexp_success_rates[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "dscrt_adpt_exp_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': dexp_l2_norms[True], 'w/o HC': hc_l2_norms[False], 'w/o Adam': adpt_l2_norms[False], 'w/o optimization': dexp_l2_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0, legend=False)
    ax1.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "dscrt_adpt_exp_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'PatchAdamS': dexp_l0_norms[True], 'w/o HC': hc_l0_norms[False], 'w/o Adam': adpt_l0_norms[False], 'w/o optimization': dexp_l0_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0)
    ax1.set_ylabel("L0 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "dscrt_adpt_exp_l0_norms.png"))

    ########################################

    tmp = plt.figure()
    df = pd.DataFrame({'K=1': l0_norms[1], 'K=3': l0_norms[3], 'K=5': l0_norms[5], 'K=7': l0_norms[7],
                       'K=10': l0_norms[10], 'C&W': l0_norms['C&W']}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0)
    ax1.set_ylabel("L0 Distortion", fontdict=dict(weight='bold'))
    plt.legend(ncol=2)
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_inst_per_dataset_l0_norms.png"))

    plt.figure()
    df = pd.DataFrame({'K=1': l2_norms[1], 'K=3': l2_norms[3], 'K=5': l2_norms[5], 'K=7': l2_norms[7],
                       'K=10': l2_norms[10], 'C&W': l2_norms['C&W']}, index=datasets_cap)
    ax2 = df.plot.bar(rot=0, legend=False)
    ax2.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_inst_per_dataset_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'K=1': success_rates[1], 'K=3': success_rates[3], 'K=5': success_rates[5], 'K=7': success_rates[7],
                       'K=10': success_rates[10], 'C&W': success_rates['C&W']}, index=datasets_cap)
    ax3 = df.plot.bar(rot=0, legend=False)
    ax3.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_inst_per_dataset_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({'K=1': attack_times[1], 'K=3': attack_times[3], 'K=5': attack_times[5], 'K=7': attack_times[7],
                       'K=10': attack_times[10], 'C&W': attack_times['C&W']}, index=datasets_cap)
    ax3 = df.plot.bar(rot=0, legend=False)
    ax3.set_ylabel("Execution Time [m]", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_inst_per_dataset_attack_times.png"))

    plt.figure()
    df = pd.DataFrame({'K=1': gen_rates[1], 'K=3': gen_rates[3], 'K=5': gen_rates[5], 'K=7': gen_rates[7],
                       'K=10': gen_rates[10], 'C&W': gen_rates['C&W']}, index=datasets_cap)
    ax5 = df.plot.bar(rot=0, legend=False)
    ax5.set_ylabel("Generalization Rate", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_inst_per_dataset_gen_rates.png"))

    plt.figure()
    df = pd.DataFrame({'K=1': gen_l2_norms[1], 'K=3': gen_l2_norms[3], 'K=5': gen_l2_norms[5], 'K=7': gen_l2_norms[7],
                       'K=10': gen_l2_norms[10], 'C&W': gen_l2_norms['C&W']}, index=datasets_cap)
    ax6 = df.plot.bar(rot=0, legend=False)
    ax6.set_ylabel("Gen. L2 Distortion", fontdict=dict(weight='bold'))
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_inst_per_dataset_gen_l2_norms.png"))

    linewidth = 5

    tmp = plt.figure()
    plt.plot(num_cnd_vec, mnist_l2_norms_per_cnd, linewidth=linewidth)
    plt.plot(num_cnd_vec, fmnist_l2_norms_per_cnd, linewidth=linewidth)
    plt.plot(num_cnd_vec, cifar10_l2_norms_per_cnd, linewidth=linewidth)
    plt.ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    plt.xlabel("# of candidates (M)", fontdict=dict(weight='bold'))
    # plt.legend(datasets_cap)
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_cnd_l2_norms.png"))

    # size = tmp.get_size_inches()
    # print(size)

    tmp = plt.figure()
    plt.plot(num_cnd_vec, mnist_success_rates_per_cnd, linewidth=linewidth)
    plt.plot(num_cnd_vec, fmnist_success_rates_per_cnd, linewidth=linewidth)
    plt.plot(num_cnd_vec, cifar10_success_rates_per_cnd, linewidth=linewidth)
    plt.ylabel("Success Rate", fontdict=dict(weight='bold'))
    plt.xlabel("# of candidates (M)", fontdict=dict(weight='bold'))
    plt.legend(datasets_cap)
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_cnd_success_rates.png"))



    tmp = plt.figure()
    plt.plot(num_cnd_vec, mnist_gen_rates_per_cnd, linewidth=linewidth)
    plt.plot(num_cnd_vec, fmnist_gen_rates_per_cnd, linewidth=linewidth)
    plt.plot(num_cnd_vec, cifar10_gen_rates_per_cnd, linewidth=linewidth)
    plt.ylabel("Generalization Rate", fontdict=dict(weight='bold'))
    plt.xlabel("# of candidates (M)", fontdict=dict(weight='bold'))
    # plt.legend(datasets_cap)
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "per_cnd_gen_rates.png"))


    font = {'size': 20}
    plt.rc('font', **font)

    tmp = plt.figure()
    df = pd.DataFrame({"(K,M)=(" + str(x) + "," + str(y) + ")" : imagenet_l0_norms[(x,y)] for
                       x in imagenet_inst for y in imagenet_cnd}, index=["ImageNet"])
    ax7 = df.plot.bar(rot=0, legend=False, figsize=(4,4))
    ax7.set_ylabel("L0 Distortion", fontdict=dict(weight='bold'))
    ax7.axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "imagenet_per_inst_per_cnd_l0_norms.png"))

    size = tmp.get_size_inches()
    print(size)

    plt.figure()
    df = pd.DataFrame({"(K,M)=(" + str(x) + "," + str(y) + ")" : imagenet_l2_norms[(x,y)] for
                       x in imagenet_inst for y in imagenet_cnd}, index=["ImageNet"])
    ax8 = df.plot.bar(rot=0, legend=False, figsize=(4,4))
    ax8.set_ylabel("L2 Distortion", fontdict=dict(weight='bold'))
    ax8.axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "imagenet_per_inst_per_cnd_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({"(K,M)=(" + str(x) + "," + str(y) + ")" : imagenet_success_rates[(x,y)] for
                       x in imagenet_inst for y in imagenet_cnd}, index=["ImageNet"])
    ax9 = df.plot.bar(rot=0, legend=False, figsize=(4,4))
    ax9.set_ylabel("Success Rate", fontdict=dict(weight='bold'))
    ax9.axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "imagenet_per_inst_per_cnd_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({"(K,M)=(" + str(x) + "," + str(y) + ")" : imagenet_attack_times[(x,y)] for
                       x in imagenet_inst for y in imagenet_cnd}, index=["ImageNet"])
    ax10 = df.plot.bar(rot=0, legend=False)
    ax10.set_ylabel("Execution Time [m]", fontdict=dict(weight='bold'))
    ax10.axes.get_xaxis().set_visible(False)
    ax10.legend(["(K,M)=(" + str(x) + "," + str(y) + ")" for x in imagenet_inst for y in imagenet_cnd], loc='center left', bbox_to_anchor=(1.1, 0.5))
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize='medium',
    #            prop={'size': 10}, borderaxespad=0.)

    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "imagenet_per_inst_per_cnd_attack_times.png"))

    # plt.figure()
    # plt.plot(num_cnd_vec, mnist_l0_norms_per_cnd)
    # plt.title("L0 Norm vs. # of candidates (MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "mnist_per_cnd_l0_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, mnist_l2_norms_per_cnd)
    # plt.title("L2 Norm vs. # of candidates (MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "mnist_per_cnd_l2_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, mnist_gen_rates_per_cnd)
    # plt.title("Generalization Rate vs. # of candidates (MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "mnist_per_cnd_gen_rates.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, mnist_gen_l2_norms_per_cnd)
    # plt.title("Generalization L2 Norm vs. # of candidates (MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "mnist_per_cnd_gen_l2_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, fmnist_l0_norms_per_cnd)
    # plt.title("L0 Norm vs. # of candidates (Fashion-MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "fmnist_per_cnd_l0_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, fmnist_l2_norms_per_cnd)
    # plt.title("L2 Norm vs. # of candidates (Fashion-MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "fmnist_per_cnd_l2_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, fmnist_gen_rates_per_cnd)
    # plt.title("Generalization Rate vs. # of candidates (Fashion-MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "fmnist_per_cnd_gen_rates.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, fmnist_gen_l2_norms_per_cnd)
    # plt.title("Generalization L2 Norm vs. # of candidates (Fashion-MNIST, 5 instructions)")
    # plt.savefig(os.path.join("plots", "fmnist_per_cnd_gen_l2_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, cifar10_l0_norms_per_cnd)
    # plt.title("L0 Norm vs. # of candidates (CIFAR10, 5 instructions)")
    # plt.savefig(os.path.join("plots", "cifar10_per_cnd_l0_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, cifar10_l2_norms_per_cnd)
    # plt.title("L2 Norm vs. # of candidates (CIFAR10, 5 instructions)")
    # plt.savefig(os.path.join("plots", "cifar10_per_cnd_l2_norms.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, cifar10_gen_rates_per_cnd)
    # plt.title("Generalization Rate vs. # of candidates (CIFAR10, 5 instructions)")
    # plt.savefig(os.path.join("plots", "cifar10_per_cnd_gen_rates.png"))
    #
    # plt.figure()
    # plt.plot(num_cnd_vec, cifar10_gen_l2_norms_per_cnd)
    # plt.title("Generalization L2 Norm vs. # of candidates (CIFAR10, 5 instructions)")
    # plt.savefig(os.path.join("plots", "cifar10_per_cnd_gen_l2_norms.png"))

    # plt.show()


def gen_noscale_exp_comp_plot():
    datasets = ["mnist", "fashion-mnist", "cifar10"]
    bin_vec = [True, False]
    exp_num_inst = 5
    exp_num_cnd = 7

    l0_norms = {x: ([None] * len(datasets)) for x in bin_vec}
    l2_norms = {x: ([None] * len(datasets)) for x in bin_vec}
    success_rates = {x: ([None] * len(datasets)) for x in bin_vec}
    attack_times = {x: ([None] * len(datasets)) for x in bin_vec}
    gen_rates = {x: ([None] * len(datasets)) for x in bin_vec}
    gen_l2_norms = {x: ([None] * len(datasets)) for x in bin_vec}

    for i, dataset in enumerate(datasets):
        for scale in bin_vec:
            if scale:
                dataset_rootdir = os.path.join("results_" + dataset, "joint_optimization", "table_exp")
            else:
                dataset_rootdir = os.path.join("noscale_exp", dataset)
            for cfg_dir in os.listdir(dataset_rootdir):
                num_inst = re.match(".*num_inst(\d+)", cfg_dir)
                num_cnd = re.match(".*cnd(\d+)", cfg_dir)
                if num_inst is not None and num_cnd is not None:
                    num_instructions = int(num_inst.group(1))
                    num_candidates = int(num_cnd.group(1))
                    if num_instructions != exp_num_inst or num_candidates != exp_num_cnd:
                        continue

                    global_stats = torch.load(os.path.join(dataset_rootdir, cfg_dir, "global_stats.pt"))
                    gen_stats = torch.load(os.path.join(dataset_rootdir, cfg_dir, "gen_stats.pt"))

                    l0_norms[scale][i] = global_stats['l0_norm']
                    l2_norms[scale][i] = global_stats['l2_norm']
                    success_rates[scale][i] = global_stats['success_rate']
                    attack_times[scale][i] = global_stats['attack_time']
                    gen_rates[scale][i] = gen_stats['prog_ave_gen_suc_rate_t']
                    gen_l2_norms[scale][i] = gen_stats['prog_ave_gen_norm_t']

                    break

    datasets_cap = ["MNIST", "Fashion-MNIST", "CIFAR-10"]

    font = {'size': 20}
    plt.rc('font', **font)

    plt.figure()
    df = pd.DataFrame({'w/ Scale': l0_norms[True], 'w/o Scale': l0_norms[False]}, index=datasets_cap)
    ax1 = df.plot.bar(rot=0)
    ax1.set_ylabel("L0 Distortion")
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "scale_exp_l0_norms.png"))

    plt.figure()
    df = pd.DataFrame({'w/ Scale': l2_norms[True], 'w/o Scale': l2_norms[False]}, index=datasets_cap)
    ax2 = df.plot.bar(rot=0, legend=False)
    ax2.set_ylabel("L2 Distortion")
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "scale_exp_l2_norms.png"))

    plt.figure()
    df = pd.DataFrame({'w/ Scale': success_rates[True], 'w/o Scale': success_rates[False]}, index=datasets_cap)
    ax3 = df.plot.bar(rot=0, legend=False)
    ax3.set_ylabel("Success Rate")
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "scale_exp_success_rates.png"))

    plt.figure()
    df = pd.DataFrame({'w/ Scale': attack_times[True], 'w/o Scale': attack_times[False]}, index=datasets_cap)
    ax3 = df.plot.bar(rot=0)
    ax3.set_ylabel("Execution Time [m]")
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "scale_exp_attack_times.png"))

    plt.figure()
    df = pd.DataFrame({'w/ Scale': gen_rates[True], 'w/o Scale': gen_rates[False]}, index=datasets_cap)
    ax5 = df.plot.bar(rot=0)
    ax5.set_ylabel("Generalization Rate")
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "scale_exp_gen_rates.png"))

    plt.figure()
    df = pd.DataFrame({'w/ Scale': gen_l2_norms[True], 'w/o Scale': gen_l2_norms[False]}, index=datasets_cap)
    ax6 = df.plot.bar(rot=0)
    ax6.set_ylabel("Gen. L2 Distortion")
    plt.gcf().set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "scale_exp_gen_l2_norms.png"))


def gen_final_plot_per_attack(rootdir, prms, device):
    titles = False
    prms.model = prms.model.to(device)
    i = 0
    for cfg_dir in os.listdir(rootdir):
        print(cfg_dir)
        num_inst = int(re.match(".*num_inst(\d+).*", cfg_dir).group(1))
        for subdir in os.listdir(os.path.join(rootdir, cfg_dir)):
            print(subdir)
            if not os.path.isdir(os.path.join(rootdir, cfg_dir, subdir)) or subdir == "failed_attacks":
                continue
            program = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'program.pt'), map_location=device)
            cw_pert = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'cw_pert.pt'), map_location=device)
            # program_stats = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'program_stats.pt'), map_location=device)
            # cw_stats = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'cw_stats.pt'), map_location=device)
            move_program_to_device(program, device)
            true_label = program.true_label.item()
            target_label = program.target_label.item()
            # run model on original image
            base_outputs = prms.model(program.source_image)
            base_exps = torch.exp(base_outputs[0])
            base_probs = base_exps / torch.sum(base_exps)
            # run program
            adv_image, l2_norm, _, probs, _ = func.apply_prog_run_model(program, prms)
            l0_norm = torch.norm(torch.flatten(adv_image - program.source_image, start_dim=1), p=0, dim=1)
            diff = (adv_image - program.source_image) / 2 + 0.5
            # run cw attack
            cw_adv_image, cw_l2_norm, _, cw_probs, _ = func.apply_pert_run_model(cw_pert, prms, program.source_image)
            cw_l0_norm = torch.norm(torch.flatten(cw_adv_image - program.source_image, start_dim=1), p=0, dim=1)
            cw_diff = (cw_adv_image - program.source_image) / 2 + 0.5

            str1 = "P(" + prms.class_names[target_label] + ") = " + str(
                round(base_probs[target_label].item() * 100, 1)) + "%" + \
                   "\nP(" + prms.class_names[true_label] + ") = " + str(
                round(base_probs[true_label].item() * 100, 1)) + "%"
            str2 = "P(" + prms.class_names[target_label] + ") = " + str(
                round(cw_probs[target_label].item() * 100, 1)) + "%" + \
                   "\nP(" + prms.class_names[true_label] + ") = " + str(
                round(cw_probs[true_label].item() * 100, 1)) + "%"
            str3 = "L2 Norm = " + str(round(cw_l2_norm.item(), 2)) + \
                   "\nL0 Norm = " + str(int(cw_l0_norm.item()))
            str4 = "P(" + prms.class_names[target_label] + ") = " + str(
                round(probs[target_label].item() * 100, 1)) + "%" + \
                   "\nP(" + prms.class_names[true_label] + ") = " + str(
                round(probs[true_label].item() * 100, 1)) + "%"
            str5 = "L2 Norm = " + str(round(l2_norm.item(), 2)) + \
                   "\nL0 Norm = " + str(int(l0_norm.item()))
            str6 = "\n"

            fontsize = 14
            if titles:
                str1 = "Original\n\n"
                str2 = "CW Attack\n\n"
                str3 = "CW Diff\n\n"
                str4 = "Our Attack\n\n"
                str5 = "Our Diff\n\n"
                str6 = "Sequence\n\n"
                fontsize = 20
                plt.rcParams["font.weight"] = "bold"
                fig = plt.figure(figsize=(18, 5))
            else:
                fig = plt.figure(figsize=(18, 4))
            # gs = fig.add_gridspec(1, 7)
            gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1],
                wspace=0.1, hspace=0.1, top=0.97, bottom=0.03, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax1.set_title(str1, fontsize=fontsize, fontweight="bold")
            imshow_wrp(program.source_image, prms.dataset)
            plt.axis('off')
            ax2 = fig.add_subplot(gs[0,1])
            ax2.set_title(str2, fontsize=fontsize, fontweight="bold")
            imshow_wrp(cw_adv_image, prms.dataset)

            plt.axis('off')
            ax3 = fig.add_subplot(gs[0,2])
            ax3.set_title(str3, fontsize=fontsize, fontweight="bold")
            imshow_wrp(cw_diff, prms.dataset)

            # tmp = torch.zeros([1,3,28,28])
            # cw_diff_alt = torch.abs(cw_adv_image - program.source_image)
            # tmp[0,0,:,:] = (cw_diff_alt > 1/256).int()
            #
            # plt.axis('off')
            # ax4 = fig.add_subplot(gs[0,3])
            # ax4.set_title(str3, fontsize=fontsize, fontweight="bold")
            # plt.imshow(tmp[0].permute([1, 2, 0]).detach().cpu(), vmin=0, vmax=1, interpolation='nearest')

            plt.axis('off')
            ax5 = fig.add_subplot(gs[0,3])
            ax5.set_title(str4, fontsize=fontsize, fontweight="bold")
            imshow_wrp(adv_image, prms.dataset)

            plt.axis('off')
            ax6 = fig.add_subplot(gs[0,4])
            ax6.set_title(str5, fontsize=fontsize, fontweight="bold")
            imshow_wrp(diff, prms.dataset)
            plt.axis('off')
            ax7 = fig.add_subplot(gs[0,5])
            ax7.set_title(str6, fontsize=fontsize, fontweight="bold")
            plot_patches(program.instructions, verbose=True)
            plt.axis('off')

            # plt.show()

            dir = os.path.join("plots_alt", "examples_" + prms.dataset,  "num_inst" + str(num_inst))
            if not os.path.exists(dir):
                os.makedirs(dir)
            plot_str = "norm" + str(round(l2_norm.item(), 1)) + "_" + prms.class_names[true_label] + "-" + prms.class_names[target_label] + "_" + str(len(program.instructions)) \
                       + "inst_" + str(i) + ".png"
            plt.savefig(os.path.join(dir, plot_str))
            plt.close(fig)
            i += 1


def gen_compact_final_plot_per_attack(rootdir, prms, device):
    titles = False
    prms.model = prms.model.to(device)
    i = 0
    for cfg_dir in os.listdir(rootdir):
        print(cfg_dir)
        num_inst = int(re.match(".*num_inst(\d+).*", cfg_dir).group(1))
        for subdir in os.listdir(os.path.join(rootdir, cfg_dir)):
            print(subdir)
            if not os.path.isdir(os.path.join(rootdir, cfg_dir, subdir)) or subdir == "failed_attacks":
                continue
            program = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'program.pt'), map_location=device)
            cw_pert = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'cw_pert.pt'), map_location=device)
            # program_stats = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'program_stats.pt'), map_location=device)
            # cw_stats = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'cw_stats.pt'), map_location=device)
            move_program_to_device(program, device)
            true_label = program.true_label.item()
            target_label = program.target_label.item()
            # run model on original image
            base_outputs = prms.model(program.source_image)
            base_exps = torch.exp(base_outputs[0])
            base_probs = base_exps / torch.sum(base_exps)
            # run program
            adv_image, l2_norm, _, probs, _ = func.apply_prog_run_model(program, prms)
            l0_norm = torch.norm(torch.flatten(adv_image - program.source_image, start_dim=1), p=0, dim=1)
            diff = (adv_image - program.source_image) / 2 + 0.5
            # run cw attack
            cw_adv_image, cw_l2_norm, _, cw_probs, _ = func.apply_pert_run_model(cw_pert, prms, program.source_image)
            cw_l0_norm = torch.norm(torch.flatten(cw_adv_image - program.source_image, start_dim=1), p=0, dim=1)
            cw_diff = (cw_adv_image - program.source_image) / 2 + 0.5

            str1 = "P(" + prms.class_names[target_label] + ") = " + str(
                round(base_probs[target_label].item() * 100, 1)) + "%" + \
                   "\nP(" + prms.class_names[true_label] + ") = " + str(
                round(base_probs[true_label].item() * 100, 1)) + "%"
            # str2 = "P(t = " + prms.class_names[target_label] + ") = " + str(
            #     round(cw_probs[target_label].item() * 100, 2)) + "%" + \
            #        "\nP(c = " + prms.class_names[true_label] + ") = " + str(
            #     round(cw_probs[true_label].item() * 100, 2)) + "%"
            # str3 = "L2 Norm = " + str(round(cw_l2_norm.item(), 2)) + \
            #        "\nL0 Norm = " + str(round(cw_l0_norm.item(), 2))
            str4 = "P(t = " + prms.class_names[target_label] + ") = " + str(
                round(probs[target_label].item() * 100, 1)) + "%" + \
                   "\nP(c = " + prms.class_names[true_label] + ") = " + str(
                round(probs[true_label].item() * 100, 1)) + "%"
            # str5 = "L2 Norm = " + str(round(l2_norm.item(), 2)) + \
            #        "\nL0 Norm = " + str(round(l0_norm.item(), 2))
            str6 = "\n"

            fontsize = 14
            if titles:
                str1 = "Original\n\n"
                # str2 = "CW Attack\n\n"
                # str3 = "CW Diff\n\n"
                str4 = "Our Attack\n\n"
                # str5 = "Our Diff\n\n"
                str6 = "Our Program\n\n"
                fontsize = 16
                plt.rcParams["font.weight"] = "bold"

            fig = plt.figure(figsize=(12,4))
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1],
                wspace=0.15, hspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95)

            ax1 = fig.add_subplot(gs[0,0])
            ax1.set_title(str1, fontsize=fontsize)
            imshow_wrp(program.source_image, prms.dataset)
            plt.axis('off')
            # ax2 = fig.add_subplot(gs[0,1])
            # ax2.set_title(str2, fontsize=fontsize)
            # imshow_wrp(cw_adv_image, prms.dataset)
            # plt.axis('off')
            # ax3 = fig.add_subplot(gs[0,2])
            # ax3.set_title(str3, fontsize=fontsize)
            # imshow_wrp(cw_diff, prms.dataset)
            # plt.axis('off')
            ax4 = fig.add_subplot(gs[0,1])
            ax4.set_title(str4, fontsize=fontsize)
            imshow_wrp(adv_image, prms.dataset)
            plt.axis('off')
            # ax5 = fig.add_subplot(gs[0,4])
            # ax5.set_title(str5, fontsize=fontsize)
            # imshow_wrp(diff, prms.dataset)
            # plt.axis('off')
            ax6 = fig.add_subplot(gs[0,2])
            ax6.set_title(str6, fontsize=fontsize)
            plot_patches(program.instructions, verbose=True)
            plt.axis('off')

            # plt.subplot_tool()
            # plt.tight_layout()
            # plt.show()
            # assert False

            dir = os.path.join("plots", "compact_examples_" + prms.dataset,  "num_inst" + str(num_inst))
            # dir = os.path.join("plots",  "num_inst" + str(num_inst))
            if not os.path.exists(dir):
                os.makedirs(dir)
            plot_str = "norm" + str(round(l2_norm.item(), 1)) + "_" + prms.class_names[true_label] + "-" + prms.class_names[target_label] + "_" + str(len(program.instructions)) \
                       + "inst_" + str(i) + ".png"
            plt.savefig(os.path.join(dir, plot_str))
            plt.close(fig)
            # assert False
            i += 1


def plot_attack(input, adv, true_label, target_label, success, norm, l0_norm, l1_norm, linf_norm, probs, orig_probs, prms, savedir=None, title="", instructions=None):
    if title != "":
        title += "\n"
    if instructions is None:
        num_plots = 2
        fig = plt.figure()
    else:
        num_plots = 3
        fig = plt.figure(figsize=(10,6))

    fig.add_subplot(1, num_plots, 1)
    imshow_wrp(input, prms.dataset)
    plt.axis('off')
    fig.add_subplot(1, num_plots, 2)
    imshow_wrp(adv, prms.dataset)
    plt.axis('off')

    if instructions is not None:
        fig.add_subplot(1, num_plots, 3)
        plot_patches(instructions)
        plt.axis('off')

    if prms.dataset == "imagenet":
        probs_str = ""
    else:
        probs_str = "\nProbs:\n["
        for i in range(probs.shape[0]):
            probs_str += "\'" + prms.class_names[i] + "\': " + str(round(orig_probs[i].item() * 100, 2)) + "%=>" + str(
                round(probs[i].item() * 100, 2)) + "%"
            if i != probs.shape[0] - 1:
                probs_str += " | "
            if i == 4:
                probs_str += "\n"
        probs_str += "]\n"

    plt.suptitle(title + "P(Target = " + prms.class_names[target_label] + ") = " + str(
                round(probs[target_label].item() * 100, 2)) + "%" +
                 "\nP(True = " + prms.class_names[true_label] + ") = " + str(
                round(probs[true_label].item() * 100, 2)) + "%\n" +
                "success = " + str(success) + "\n" +
                 r"$\bf{L2\_norm = " + str(round(norm, 2)) + "}$, L0_norm = " + str(int(l0_norm)) +
                 ", L1_norm = " + str(round(l1_norm, 2)) + ", Linf_norm = " + str(round(linf_norm, 2)) +
                 probs_str)
    if savedir is None or not os.path.exists(savedir):
        plt.show()
    else:
        tmp_str = str(true_label) + "=>" + str(target_label)
        fig.savefig(os.path.join(savedir, tmp_str + ".png"))
    plt.close(fig)


def plot_attack_cw(input, adv, cw_attack, cw_success, cw_l2_norm, cw_l0_norm, cw_l1_norm, cw_linf_norm, true_label, target_label, success, norm, l0_norm, l1_norm, linf_norm, probs, orig_probs, prms, savedir=None, title="", instructions=None):
    if title != "":
        title += "\n"
    if instructions is None:
        num_plots = 2
        fig = plt.figure()
    else:
        num_plots = 4
        fig = plt.figure(figsize=(12,6))

    fig.add_subplot(1, num_plots, 1)
    imshow_wrp(input, prms.dataset)
    plt.axis('off')
    fig.add_subplot(1, num_plots, 2)
    imshow_wrp(cw_attack, prms.dataset)
    plt.axis('off')
    fig.add_subplot(1, num_plots, 3)
    imshow_wrp(adv, prms.dataset)
    plt.axis('off')

    if instructions is not None:
        fig.add_subplot(1, num_plots, 4)
        plot_patches(instructions)
        plt.axis('off')

    if prms.dataset == "imagenet":
        probs_str = ""
    else:
        probs_str = "\nProbs:\n["
        for i in range(probs.shape[0]):
            probs_str += "\'" + prms.class_names[i] + "\': " + str(round(orig_probs[i].item() * 100, 2)) + "%=>" + str(
                round(probs[i].item() * 100, 2)) + "%"
            if i != probs.shape[0] - 1:
                probs_str += " | "
            if i == 4:
                probs_str += "\n"
        probs_str += "]\n"

    plt.suptitle(title + "P(Target = " + prms.class_names[target_label] + ") = " + str(
                round(probs[target_label].item() * 100, 2)) + "%" +
                 "\nP(True = " + prms.class_names[true_label] + ") = " + str(
                round(probs[true_label].item() * 100, 2)) + "%\n" +
                "success = " + str(success) + ", cw_success = " + str(cw_success) + "\n" +
                 r"$\bf{L2\_norm = " + str(round(norm, 2)) + "}$, L0_norm = " + str(int(l0_norm)) +
                 ", L1_norm = " + str(round(l1_norm, 2)) + ", Linf_norm = " + str(round(linf_norm, 2)) +
                 "\ncw_L2_norm = " + str(round(cw_l2_norm, 2)) + ", cw_L0_norm = " + str(round(cw_l0_norm, 2)) +
                 ", cw_L1_norm = " + str(round(cw_l1_norm, 2)) + ", cw_Linf_norm = " + str(round(cw_linf_norm, 2)) +
                 probs_str)
    if savedir is None or not os.path.exists(savedir):
        plt.show()
    else:
        tmp_str = str(true_label) + "=>" + str(target_label)
        fig.savefig(os.path.join(savedir, tmp_str + ".png"))
    plt.close(fig)


def plot_loss_norm_success(num_itr, loss_values, norm_values, success_values, best_itr, inst_itr_markers, savedir=None):
    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(3, 1, 1)
    plt.plot(range(num_itr), loss_values)
    plt.ylabel('loss')
    for m in inst_itr_markers:
        plt.axvline(x=m, color='k', linestyle='--')
    plt.axvline(x=best_itr, color='g')
    fig.add_subplot(3, 1, 2)
    plt.plot(range(num_itr), norm_values)
    plt.ylabel('norm')
    for m in inst_itr_markers:
        plt.axvline(x=m, color='k', linestyle='--')
    plt.axvline(x=best_itr, color='g')
    fig.add_subplot(3, 1, 3)
    plt.plot(range(num_itr), success_values)
    plt.ylabel('attack success')
    plt.yticks([0,1])
    plt.xlabel('iterations')
    for m in inst_itr_markers:
        plt.axvline(x=m, color='k', linestyle='--')
    plt.axvline(x=best_itr, color='g')

    plt.suptitle("best iteration " + str(best_itr+1))

    if savedir is None or not os.path.exists(savedir):
        plt.show()
    else:
        fig.savefig(os.path.join(savedir, "optim.png"))
    plt.close(fig)

