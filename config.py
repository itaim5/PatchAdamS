import copy
import re
import pandas as pd
from core import *


class Config:
    def __init__(self, dir):
        self.dir = dir
        self.parse_dir_name(dir)

    def parse_dir_name(self, dir):
        attr_list = ["num_inst", "clr", "dlr", "citr", "optitr", "cnd", "snr", "lpt", "lpp", "itrInst", "loadBase",
                     "stds", "sizes", "betas", "num_attacks", "norm"]
        typelist = [int, float, float, int, int, int, float, float, int, bool, bool, list, list, list, int, float]
        for i, attr in enumerate(attr_list):
            m = re.match(".*" + attr + "([-+]?\d*\.\d+|\d+)(?:_|$)", dir)
            if m:
                val = m.group(1)
                if typelist[i] == bool:
                    val = int(val)
                setattr(self, attr, typelist[i](val))
            else:
                if attr == "snr":
                    setattr(self, attr, typelist[i](1))
                elif attr == "cnd":
                    setattr(self, attr, typelist[i](1))
                else:
                    setattr(self, attr, None)


def attribute_dicts_eq(d1, d2, attr_to_ignore=None):
    attr_list_to_compare = ["num_inst", "clr", "dlr", "citr", "optitr", "cnd", "snr", "lpt", "lpp", "itrInst",
                            "loadBase","stds", "sizes", "betas"]
    if attr_to_ignore is not None:
        attr_list_to_compare.remove(attr_to_ignore)
    for attr in attr_list_to_compare:
        if d1[attr] != d2[attr]:
            return False
    return True


def create_attr_based_groups(config_list, attr):
    dicts_wo_attr = {}
    # pop the anchor attribute for all configs
    for i, cfg in enumerate(config_list):
        dict_wo_attr = copy.deepcopy(cfg.__dict__)
        dict_wo_attr.pop(attr)
        dicts_wo_attr[i] = dict_wo_attr
    # find groups of identical configs, which differs only in the anchor attr
    groups = []
    for i in range(len(config_list)):
        d1 = dicts_wo_attr.pop(i, None)
        if d1 is None:
            continue
        group = [i]
        for j in range(len(config_list)):
            d2 = dicts_wo_attr.get(j)
            if d2 is None:
                continue
            assert i != j
            if attribute_dicts_eq(d1, d2, attr):
                group.append(j)
                del dicts_wo_attr[j]
        groups.append(group)
    # sort each group by attribute
    for i, gr in enumerate(groups):
        new_gr = sorted(gr, key=lambda i: getattr(config_list[i], attr))
        groups[i] = new_gr
    return groups


if __name__ == '__main__':
    rootdir = os.path.join("results_cifar10", "joint_optimization")
    cfgs = []
    # create Config structures for all configurations
    for _, dirs, _ in os.walk(rootdir):
        for subdir in dirs:
            num_inst = re.match(".*(num_inst).*", subdir)
            num_attacks = re.match(".*(num_attacks).*", subdir)
            if num_attacks and num_inst:
                print(subdir)
                cfg = Config(subdir)
                cfgs.append(cfg)
        break

    attr = 'clr'
    citr_groups = create_attr_based_groups(cfgs, attr)
    for i, g in enumerate(citr_groups):
        # ignore groups with only 1 member
        if len(g) == 1:
            continue
        # group with same cfg besides attr
        df = pd.DataFrame(columns=[attr, 'src=>trgt', 'norm'])
        for j in g:
            cfg_dir = cfgs[j].dir
            attr_val = getattr(cfgs[j], attr)
            for _, dirs, _ in os.walk(os.path.join(rootdir, cfg_dir)):
                for subdir in dirs:
                    # failed attacks
                    if subdir == "failed_attacks":
                        for _, inner_dirs, _ in os.walk(os.path.join(rootdir, cfg_dir, subdir)):
                            for inner_subdir in inner_dirs:
                                src_trgt = inner_subdir[:4]
                                df = df.append({attr: attr_val, 'src=>trgt': src_trgt, 'norm': 10.}, ignore_index=True)
                            break
                    # successful attacks
                    else:
                        src_trgt = subdir[:4]
                        norm = float(re.match(".*norm(\d+\.\d+)", subdir).group(1))
                        # print(str(attr_val), src_trgt, str(norm))
                        df = df.append({attr: attr_val, 'src=>trgt': src_trgt, 'norm': norm}, ignore_index=True)
                break

        cfg_str = re.sub(attr + "([-+]?\d*\.\d+|\d+)(?:_|$)", "", cfg_dir)
        cfg_str = re.sub("_num_attacks.*", "", cfg_str)

        ag = df.groupby(['src=>trgt', attr]).mean().unstack()
        ag.columns = ag.columns.droplevel()
        ag.plot(kind='bar', figsize=(14,5), ylabel='norm', title=cfg_str)
        # plt.show()
        plt.savefig(os.path.join(rootdir, attr + " - " + cfg_str + ".png"))