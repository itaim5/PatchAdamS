from utils import *


class PatchInitCfg:
    def __init__(self, std_scale, std_shift, min_size, max_size):
        self.std_scale = std_scale
        self.std_shift = std_shift
        self.min_size = min_size
        self.max_size = max_size


class Program:
    def __init__(self, true_label, target_label, source_image, num_classes, device, s_bounds=None, e_bounds=None):
        self.true_label = true_label
        self.target_label = target_label
        self.instructions = []
        self.source_image = source_image
        self.target_one_hot = torch.zeros([num_classes], device=device)
        self.target_one_hot[target_label] = 1
        self.num_classes = num_classes
        self.device = device

        self.s_bounds = s_bounds if s_bounds is not None else [0, 0, 0]
        self.e_bounds = e_bounds if e_bounds is not None else list(self.source_image.shape)

        self.image_mask = (source_image > 1).int()

    @classmethod
    def copy(cls, program):
        new_prog = cls(program.true_label, program.target_label, program.source_image.detach().clone(),
                       program.num_classes, program.device, program.s_bounds, program.e_bounds)
        new_prog.instructions = [PatchInstruction(Patch([t.detach().clone() for t in inst.patch.start_ind],
                                                        [t.detach().clone() for t in inst.patch.patch_shape],
                                                        inst.patch.image_shape, inst.patch.device,
                                                        program.s_bounds, program.e_bounds),
                                                        inst.scale, inst.shift, inst.device) for inst
                                                                                             in program.instructions]
        return new_prog

    @classmethod
    def restore_from_compact(cls, compact, true_label, target_label, source_image, num_classes, device, s_bounds, e_bounds):
        new_prog = cls(true_label, target_label, source_image, num_classes, device, s_bounds, e_bounds)
        image_shape = torch.Size([x for x in source_image.shape[1:]])
        for d in compact:
            start_ind = [torch.tensor([x], device=device, requires_grad=False, dtype=torch.float) for x in
                           d['start_ind']]
            patch_shape = [torch.tensor([x], device=device, requires_grad=False, dtype=torch.float) for x in
                           d['patch_shape']]
            patch = Patch(start_ind, patch_shape, image_shape, device, s_bounds, e_bounds)
            inst = PatchInstruction(patch, d['scale'], d['shift'], device)
            new_prog.instructions.append(inst)
        return new_prog

    def save_compact(self):
        compact = []
        for inst in self.instructions:
            d = {'scale': [inst.scale[i].item() for i in range(inst.scale.shape[0])],
                 'shift': [inst.shift[i].item() for i in range(inst.shift.shape[0])],
                 'start_ind': [x.item() for x in inst.patch.start_ind],
                 'patch_shape': [x.item() for x in inst.patch.patch_shape]}
            compact.append(d)
        return compact

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def remove_instruction(self, instruction):
        self.instructions.remove(instruction)

    def apply_program(self, input=None):
        if input is None:
            input = self.source_image
        output_image = torch.clone(input)
        for inst in self.instructions:
            output_image[0] = inst.apply(output_image[0])
        return output_image

    def __str__(self):
        desc = ''
        desc += "### Program ###\n"
        for inst in self.instructions:
            desc += str(inst) + "\n"
        desc += "\n"
        return desc


class Patch:
    def __init__(self, start_ind, patch_shape, image_shape, device, s_bounds=None, e_bounds=None):
        self.device = device
        self.start_ind = start_ind
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.size = self.patch_shape[1] * self.patch_shape[2]

        self.s_bounds = s_bounds if s_bounds is not None else [0,0,0]
        self.e_bounds = e_bounds if e_bounds is not None else self.image_shape

        rnd_start_ind = [round(x.item()) for x in self.start_ind]
        rnd_patch_shape = [round(x.item()) for x in self.patch_shape]

        self.patch_mask = torch.zeros(image_shape, device=self.device, dtype=torch.int)
        self.patch_mask[rnd_start_ind[0]: (rnd_start_ind[0] + rnd_patch_shape[0]),
                        rnd_start_ind[1]: (rnd_start_ind[1] + rnd_patch_shape[1]),
                        rnd_start_ind[2]: (rnd_start_ind[2] + rnd_patch_shape[2])] = 1
        self.patch_not_mask = torch.ones(image_shape, device=self.device) - self.patch_mask

    def is_legit(self):
        # print("bounds", self.s_bounds, self.e_bounds)
        # print([round(x.item()) for x in self.start_ind ], [round(x.item()) + round(y.item()) for x,y in zip(self.start_ind, self.patch_shape)])
        for (start, patch_dim, s_bound, e_bound) in zip(self.start_ind, self.patch_shape, self.s_bounds, self.e_bounds):
            if round(start.item()) < s_bound or round(patch_dim.item()) < 1 or \
                    round(start.item()) + round(patch_dim.item()) > e_bound:
                return False
        return True

    def update_masks(self):
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.patch_mask = torch.zeros(self.image_shape, device=device, dtype=torch.int)
        rnd_start_ind = [round(x.item()) for x in self.start_ind]
        rnd_patch_shape = [round(x.item()) for x in self.patch_shape]
        self.patch_mask[rnd_start_ind[0]: (rnd_start_ind[0] + rnd_patch_shape[0]),
                        rnd_start_ind[1]: (rnd_start_ind[1] + rnd_patch_shape[1]),
                        rnd_start_ind[2]: (rnd_start_ind[2] + rnd_patch_shape[2])] = 1
        self.patch_not_mask = torch.ones(self.image_shape, device=device) - self.patch_mask

    def eq(self, other):
        return self.start_ind == other.start_ind and self.patch_shape == other.patch_shape

    def __str__(self):
        return "Patch: shape - " + str([round(x.item()) for x in self.patch_shape]) + ", start_ind - " + \
               str([round(x.item()) for x in self.start_ind])


class Instruction:
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError

    def apply(self, image):
        raise NotImplementedError


class PatchInstruction(Instruction):
    def __init__(self, patch, scale, shift, device):
        self.device = device
        self.name = "PatchInstruction"
        self.patch = patch
        if isinstance(scale, torch.Tensor) and isinstance(shift, torch.Tensor):
            self.scale = scale.detach().clone()
            self.shift = shift.detach().clone()
            self.scale.requires_grad = True
            self.shift.requires_grad = True
        elif isinstance(scale, list) and isinstance(shift, list) and len(scale) == 3 and len(shift) == 3:
            self.scale = torch.tensor(scale, device=self.device).unsqueeze(1).unsqueeze(2).detach().clone()
            self.shift = torch.tensor(shift, device=self.device).unsqueeze(1).unsqueeze(2).detach().clone()
            self.scale.requires_grad = True
            self.shift.requires_grad = True
        elif isinstance(scale, list) and isinstance(shift, list) and len(scale) == 1 and len(shift) == 1:
            self.scale = torch.tensor(scale, device=self.device, requires_grad=True)
            self.shift = torch.tensor(shift, device=self.device, requires_grad=True)
        else:
            self.scale = torch.tensor([scale], device=self.device, requires_grad=True)
            self.shift = torch.tensor([shift], device=self.device, requires_grad=True)

    def __str__(self):
        if self.scale.shape[0] == 3 and self.shift.shape[0] == 3:
            scale = [round(self.scale[i,0,0].item(), 2) for i in range(3)]
            shift = [round(self.shift[i,0,0].item(), 2) for i in range(3)]
        else:
            scale = round(self.scale.item(), 2)
            shift = round(self.shift.item(), 2)
        return str(self.patch) + ", scale: " + str(scale) + ", shift: " + str(shift)

    def params(self):
        return [self.patch, self.scale, self.shift]

    def apply(self, image):
        assert (image.shape == self.patch.image_shape)
        return torch.clone(image) * (self.patch.patch_not_mask + self.patch.patch_mask * self.scale) + \
                self.patch.patch_mask * self.shift


    @staticmethod
    def generate(patch_init_cfg, prms, device, s_bounds, e_bounds):
        patch, scale, shift = PatchInstruction.generate_params(torch.Size(prms.image_shape), patch_init_cfg, prms,
                                                               device, s_bounds, e_bounds)
        return PatchInstruction(patch, scale, shift, device)

    @staticmethod
    def generate_params(image_shape, patch_init_cfg, prms, device, s_bounds, e_bounds):
        dim_num = len(image_shape)
        assert (dim_num == 3)
        # init scale and shift (currently (1,0))
        if prms.dataset == "mnist" or prms.dataset == "fashion-mnist":
            scale = np.random.normal(1., patch_init_cfg.std_scale)
            shift = np.random.normal(0., patch_init_cfg.std_shift)
        elif prms.dataset == "cifar10" or prms.dataset == "imagenet":
            scale = np.random.normal(1., patch_init_cfg.std_scale, 3).tolist()
            shift = np.random.normal(0., patch_init_cfg.std_shift, 3).tolist()


        # generate random patch
        patch_shape_list = [image_shape[0]] + np.random.randint(patch_init_cfg.min_size, patch_init_cfg.max_size+1, 2).tolist()
        start_index_list = [0] + [np.random.randint(s_bounds[i], e_bounds[i]-patch_shape_list[i]+1) for i in range(1,3)]

        patch_shape = [torch.tensor([x], device=device, requires_grad=False, dtype=torch.float) for x in patch_shape_list]
        start_index = [torch.tensor([x], device=device, requires_grad=False, dtype=torch.float) for x in start_index_list]

        patch = Patch(start_index, patch_shape, image_shape, device, s_bounds, e_bounds)
        assert (patch.is_legit())
        return patch, scale, shift
