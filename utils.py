import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import math
import re
import pandas as pd

ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0


def hls_to_rgb(h, l, s):
    if s == 0.0:
        return l, l, l
    if l <= 0.5:
        m2 = l * (1.0+s)
    else:
        m2 = l+s-(l*s)
    m1 = 2.0*l - m2
    return _v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD)


def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < ONE_SIXTH:
        return m1 + (m2-m1)*hue*6.0
    if hue < 0.5:
        return m2
    if hue < TWO_THIRD:
        return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
    return m1


def prepare_class_sets(test_data, prms, kwargs):
    test_class_loaders = []
    val_class_sets = []
    test_data.targets = torch.LongTensor(test_data.targets)
    class_val_set_size = math.floor(prms.val_set_size / prms.num_classes)
    for c in range(prms.num_classes):
        idx = [x.item() for x in torch.nonzero(test_data.targets == c, as_tuple=False).flatten()]
        # random.shuffle(idx)
        # extract small set for validation
        if class_val_set_size != 0:
            class_val_data = torch.utils.data.Subset(test_data, idx[:class_val_set_size])
            class_val_set = torch.stack([t[0] for t in class_val_data])
            val_class_sets.append(class_val_set)
        # create test set and loader
        class_test_data = torch.utils.data.Subset(test_data, idx[class_val_set_size:])
        class_loader = torch.utils.data.DataLoader(class_test_data, shuffle=False, **kwargs)
        test_class_loaders.append(class_loader)
        print("class %d datasets - test_set_size=%d, val_set_size=%d" % (c, len(class_loader.dataset),
                                                              class_val_set.shape[0] if class_val_set_size else 0))
    return test_class_loaders, val_class_sets


def prepare_examples(class_loaders, prms):
    if os.path.isfile(prms.dataset + "_examples.pt"):
        examples = torch.load(prms.dataset + "_examples.pt")
    else:
        examples = torch.zeros(prms.num_classes, prms.num_examples, *prms.image_shape)
        for c in range(prms.num_classes):
            class_batch = next(iter(class_loaders[c]))[0]
            assert(prms.num_examples < class_batch.shape[0])
            examples[c] = class_batch[:prms.num_examples]
        examples = examples
        torch.save(examples, prms.dataset + "_examples.pt")
    if not os.path.isfile(prms.dataset + "_examples.png"):
        examples_fig = plt.figure(figsize=(prms.num_examples, prms.num_classes))
        for i in range(examples.shape[0]):
            for j in range(examples.shape[1]):
                examples_fig.add_subplot(prms.num_examples, prms.num_classes, j * examples.shape[0] + i + 1)
                if prms.dataset == "mnist" or prms.dataset == "fashion-mnist":
                    plt.imshow(examples[j, i, 0].cpu(), "gray", interpolation='nearest')
                elif prms.dataset == "cifar10":
                    plt.imshow(examples[j, i].permute([1,2,0]).cpu(), interpolation='nearest')
                else:
                    raise NotImplementedError
                plt.axis('off')
                plt.subplots_adjust(wspace=0.02, hspace=0.02)
        examples_fig.savefig(prms.dataset + "_examples")
    return examples


def calc_base_accuracies(test_class_loaders, val_class_sets, prms, device):
    # evaluate test set to set base accuracies
    for c in range(prms.num_classes):
        num_class_samples = len(test_class_loaders[c].dataset)
        class_corrects = 0.0
        # evaluate validation set
        x_batch = val_class_sets[c].to(device)
        output = prms.model(x_batch)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        val_class_corrects = torch.sum(pred == c)
        prms.val_base_accuracies[c] = round((val_class_corrects.float() / x_batch.shape[0]).item(), 3)
        # evaluate test set
        for i, (data, target) in enumerate(test_class_loaders[c]):
            x_batch, y_batch = data.to(device), target
            assert((y_batch == c).all())
            output = prms.run_model(x_batch)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            class_corrects += torch.sum(pred == c).item()
        class_test_acc = (class_corrects / num_class_samples)
        prms.test_base_accuracies[c] = round(class_test_acc, 3)


def gen_examples_plot(class_examples, modified_examples, prms):
    ex_output = F.softmax(prms.model(modified_examples),1)

    fig = plt.figure(figsize=(15, 4))
    plt.suptitle('class modified examples vs. originals')
    if prms.dataset == "mnist":
        for e in range(prms.num_examples):
            fig.add_subplot(3, prms.num_examples, e + 1)
            plt.imshow(class_examples[e, 0].cpu(), "gray", interpolation='nearest')
            plt.axis('off')
            fig.add_subplot(3, prms.num_examples, e + prms.num_examples + 1)
            plt.imshow(modified_examples[e, 0].cpu(), "gray", interpolation='nearest')
            plt.axis('off')
            fig.add_subplot(3, prms.num_examples, e + 2 * prms.num_examples + 1)
            plt.bar(np.arange(prms.num_classes), ex_output[e, :].cpu().detach().numpy())
            plt.xticks(np.arange(prms.num_classes))
    elif prms.dataset == "cifar10":
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for e in range(prms.num_examples):
            fig.add_subplot(3, prms.num_examples, e + 1)
            plt.imshow(class_examples[e].permute([1,2,0]).cpu(), interpolation='nearest')
            plt.axis('off')
            fig.add_subplot(3, prms.num_examples, e + prms.num_examples + 1)
            plt.imshow(modified_examples[e].permute([1,2,0]).cpu(), interpolation='nearest')
            plt.axis('off')
            fig.add_subplot(3, prms.num_examples, e + 2 * prms.num_examples + 1)
            plt.bar(np.arange(prms.num_classes), ex_output[e, :].cpu().detach().numpy())
            plt.xticks(np.arange(prms.num_classes), class_names, rotation='vertical')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def move_program_to_device(program, device):
    program.true_label = program.true_label.to(device)
    program.target_label = program.target_label.to(device)
    program.source_image = program.source_image.to(device)
    program.target_one_hot = program.target_one_hot.to(device)
    for inst in program.instructions:
        inst.scale = inst.scale.to(device).detach().requires_grad_(True)
        inst.shift = inst.shift.to(device).detach().requires_grad_(True)
        inst.patch.patch_mask = inst.patch.patch_mask.to(device).detach()
        inst.patch.patch_not_mask = inst.patch.patch_not_mask.to(device).detach()
        inst.device = device
        inst.patch.device = device


def get_n_good_samples_per_class(data, n, prms, classes):
    tmp = [n] * len(classes)
    sample_indices = []
    sample_targets = []
    for i, t in enumerate(data.targets):
        if t.item() in classes:
            ind = classes.index(t)
            if tmp[ind] > 0:
                outputs = prms.model(data[i][0].unsqueeze(0))
                pred = torch.max(outputs[0], 0)[1]
                # verify model prediction
                assert t == data[i][1]
                if pred.item() == t:
                    sample_indices.append(i)
                    sample_targets.append(t.item())
                    tmp[ind] -= 1
    for t in tmp:
        assert t == 0
    return sample_indices, sample_targets


def get_dir_path(root_dir, true_label, target_label, norm, prms, failed=False, prob_diff=None):
    tmp = 0
    while True:
        child_dir = str(prms.class_names[true_label]) + "=>" + str(prms.class_names[target_label]) + \
                    ".norm" + str(round(norm, 2))
        if tmp != 0:
            child_dir += "_" + str(tmp)
        if failed:
            assert prob_diff is not None
            child_dir += ".prob_diff" + str(round(prob_diff*100, 1)) + "%"
            dir_path = os.path.join(root_dir, "failed_attacks", child_dir)
        else:
            dir_path = os.path.join(root_dir, child_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            break
        tmp += 1
    return dir_path


def print_program_with_orig(program, orig_instructions):
    desc = ''
    desc += "### Program ###\n"
    for i, inst in enumerate(program.instructions):
        desc += "#" + str(i+1) + "#\n"
        desc += str(orig_instructions[i]) + "\n" + str(inst) + "\n\n"
    desc += "\n"
    return desc


def get_average_attack_time(rootdir):
    for _, dirs, _ in os.walk(rootdir):
        for i, subdir in enumerate(dirs):
            num_inst = re.match(".*(num_inst).*", subdir)
            num_attacks = re.match(".*(num_attacks).*", subdir)
            time = re.match(".*(time).*", subdir)
            if num_inst is not None and num_attacks is not None and time is None:
                log_file = os.path.join(rootdir, subdir, 'log.txt')
                times = []
                with open(log_file, 'r') as f:
                    for line in f:
                        time = re.match(".* (\d+) minutes", line)
                        if time is not None:
                            time = int(time.group(1))
                            times.append(time)
                ave_time = sum(times) / len(times)
                os.rename(os.path.join(rootdir,subdir), os.path.join(rootdir,subdir) + "_time" + str(round(ave_time, 1)))
        break


def calc_global_stats(rootdir):
    # dataset = re.match(".*results_([a-zA-Z0-9-]+).*", rootdir).group(1)

    for cfg_dir in os.listdir(rootdir):
        # if os.path.exists(os.path.join(rootdir, cfg_dir, "global_stats.pt")):
        #     continue
        print(cfg_dir)
        cnt = 0
        failure_cnt = 0
        num_inst_cnt = 0.
        l0_norm_cnt = 0.
        l1_norm_cnt = 0.
        l2_norm_cnt = 0.
        linf_norm_cnt = 0.
        suc_cnt = 0
        time_cnt = 0.
        prll_cnt = 0.
        # cw_l0_norm_cnt = 0.
        # cw_l1_norm_cnt = 0.
        # cw_l2_norm_cnt = 0.
        # cw_linf_norm_cnt = 0.
        # cw_suc_cnt = 0
        # cw_time_cnt = 0.
        # cw_prll_cnt = 0.
        l2_norms = []
        cw_l2_norms = []
        for subdir in os.listdir(os.path.join(rootdir, cfg_dir)):
            if not os.path.isdir(os.path.join(rootdir, cfg_dir, subdir)):
                continue
            if subdir == "failed_attacks":
                for _ in os.listdir(os.path.join(rootdir, cfg_dir, subdir)):
                    failure_cnt += 1
                continue
            program_stats = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'program_stats.pt'))
            # cw_stats = torch.load(os.path.join(rootdir, cfg_dir, subdir, 'cw_stats.pt'))

            cnt += 1
            num_inst_cnt += int(program_stats['num_instructions'])
            l0_norm_cnt += int(program_stats['l0_norm'])
            l1_norm_cnt += float(program_stats['l1_norm'])
            l2_norm_cnt += float(program_stats['l2_norm'])
            l2_norms.append(float(program_stats['l2_norm']))
            linf_norm_cnt += float(program_stats['linf_norm'])
            suc_cnt += int(program_stats['success'])
            time_cnt += float(program_stats['attack_time'])
            prll_cnt += int(program_stats['parallel'])
            # cw_l0_norm_cnt += int(cw_stats['l0_norm'])
            # cw_l1_norm_cnt += float(cw_stats['l1_norm'])
            # cw_l2_norm_cnt += float(cw_stats['l2_norm'])
            # cw_l2_norms.append(float(cw_stats['l2_norm']))
            # cw_linf_norm_cnt += float(cw_stats['linf_norm'])
            # cw_suc_cnt += int(cw_stats['success'])
            # cw_time_cnt += float(cw_stats['attack_time'])
            # cw_prll_cnt += int(cw_stats['parallel'])

        true_label = program_stats['true_label']
        target_label = program_stats['target_label']

        num_inst = round(num_inst_cnt / cnt, 2)
        l0_norm = round(l0_norm_cnt / cnt, 2)
        l1_norm = round(l1_norm_cnt / cnt, 2)
        l2_norm = round(l2_norm_cnt / cnt, 2)
        linf_norm = round(linf_norm_cnt / cnt, 2)
        success_rate = round(suc_cnt * 100 / (cnt + failure_cnt), 1)
        time = round(time_cnt / cnt, 2)
        prll = round(prll_cnt / cnt, 2)
        # cw_l0_norm = round(cw_l0_norm_cnt / cnt, 2)
        # cw_l1_norm = round(cw_l1_norm_cnt / cnt, 2)
        # cw_l2_norm = round(cw_l2_norm_cnt / cnt, 2)
        # cw_linf_norm = round(cw_linf_norm_cnt / cnt, 2)
        # cw_success_rate = round(cw_suc_cnt * 100 / cnt, 1)
        # cw_time = round(cw_time_cnt / cnt, 2)
        # cw_prll = round(cw_prll_cnt / cnt, 2)

        d = {'true_label': true_label, 'target_label': target_label, 'success_rate': success_rate,
             'num_instructions' : num_inst, 'l0_norm': l0_norm, 'l1_norm': l1_norm, 'l2_norm': l2_norm,
             'linf_norm': linf_norm, 'attack_time': time, 'parallel': prll}
             # 'cw_l0_norm': cw_l0_norm, 'cw_l1_norm': cw_l1_norm, 'cw_l2_norm': cw_l2_norm,
             # 'cw_linf_norm': cw_linf_norm, 'cw_attack_time': cw_time, 'cw_parallel': cw_prll}
        print(d)
        torch.save(d, os.path.join(rootdir, cfg_dir, "global_stats.pt"))
        with open(os.path.join(rootdir, cfg_dir, 'global_stats.txt'), 'w') as f:
            print(str(d), file=f)

        plt.figure()
        df = pd.DataFrame(columns=['l2_norm'])
        for i, x in enumerate(l2_norms):
            df = df.append({'l2_norm': x}, ignore_index=True)
        df = df.sort_values('l2_norm')
        ax = df.plot(kind='bar', figsize=(15,5), ylabel='L2 Norm', xticks=None, legend=False)
        ax.set_xticklabels('')
        plt.savefig(os.path.join(rootdir, cfg_dir, "l2_norm_bars.png"))

        fig = plt.figure()
        plt.hist(l2_norms, bins=10)
        plt.xlabel('L2 Norm')
        plt.ylabel('# of attacks')
        fig.savefig(os.path.join(rootdir, cfg_dir, "l2_norm_hist.png"))

        fig = plt.figure()
        plt.hist(cw_l2_norms, bins=10)
        plt.xlabel('CW L2 Norm')
        plt.ylabel('# of attacks')
        fig.savefig(os.path.join(rootdir, cfg_dir, "cw_l2_norm_hist.png"))

