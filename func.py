import torch.optim as optim
from config import *
from discrete_adam import DiscreteAdam
import advertorch
from color import ciede2000_diff, rgb2lab, lab2rgb
import vis
import time
from filelock import FileLock
from instructions import Program, Patch, PatchInstruction


def apply_prog(program, input=None):
    output_image = program.apply_program(input)
    # assert (0 <= output_image).all() and (output_image <= 1).all()
    # output_image = torch.round(output_image * 256.) / 256.
    return torch.clamp(output_image, 0, 1)


def apply_prog_cielab(program, input=None):
    lab_low_bnds = torch.tensor([[[[0.]],[[-200.]],[[-200.]]]], device=input.device)
    lab_upp_bnds = torch.tensor([[[[100.]],[[200.]],[[200.]]]], device=input.device)

    # convert to CIELAB
    input_lab = rgb2lab(input, input.device)
    # apply program on CIELAB space
    adv_lab_1 = program.apply_program(input_lab)
    adv_lab = torch.max(torch.min(adv_lab_1, lab_upp_bnds), lab_low_bnds)

    if torch.count_nonzero(torch.isnan(adv_lab)) != 0:
        print(str(program))
        print("max_i0", torch.max(input[:,0,:,:]), "min_i0", torch.min(input[:,0,:,:]))
        print("max_i1", torch.max(input[:,1,:,:]), "min_i1", torch.min(input[:,1,:,:]))
        print("max_i2", torch.max(input[:,2,:,:]), "min_i2", torch.min(input[:,2,:,:]))
        print("max_ilab0", torch.max(input_lab[:,0,:,:]), "min_ilab0", torch.min(input_lab[:,0,:,:]))
        print("max_ilab1", torch.max(input_lab[:,1,:,:]), "min_ilab1", torch.min(input_lab[:,1,:,:]))
        print("max_ilab2", torch.max(input_lab[:,2,:,:]), "min_ilab2", torch.min(input_lab[:,2,:,:]))
        print("max_alab0", torch.max(adv_lab_1[:,0,:,:]), "min_alab0", torch.min(adv_lab_1[:,0,:,:]))
        print("max_alab1", torch.max(adv_lab_1[:,1,:,:]), "min_alab1", torch.min(adv_lab_1[:,1,:,:]))
        print("max_alab2", torch.max(adv_lab_1[:,2,:,:]), "min_alab2", torch.min(adv_lab_1[:,2,:,:]))
        print("max_alab0", torch.max(adv_lab[:,0,:,:]), "min_alab0", torch.min(adv_lab[:,0,:,:]))
        print("max_alab1", torch.max(adv_lab[:,1,:,:]), "min_alab1", torch.min(adv_lab[:,1,:,:]))
        print("max_alab2", torch.max(adv_lab[:,2,:,:]), "min_alab2", torch.min(adv_lab[:,2,:,:]))
        assert False

    # convert back to RGB
    adv_rgb = lab2rgb(adv_lab, input.device)

    # identity sanity check
    rgb_idnty = lab2rgb(input_lab, input.device)
    idnty_norm = torch.norm(torch.flatten(input - rgb_idnty, start_dim=1), p=2, dim=1)
    if idnty_norm > 0.1:
        print(idnty_norm.item())
    assert idnty_norm < 0.1

    return torch.clamp(adv_rgb, 0, 1)


def apply_pert(pert, input):
    output_image = input + pert
    # assert (0 <= output_image).all() and (output_image <= 1).all()
    return torch.clamp(output_image, 0, 1)


def apply_pert_run_model(pert, prms, input):
    adv_image = apply_pert(pert, input)
    norm = torch.norm(torch.flatten(adv_image - input, start_dim=1), p=2, dim=1)
    outputs = prms.model(adv_image)
    pred = torch.max(outputs[0], 0)[1]
    exps = torch.exp(outputs[0])
    probs = exps / torch.sum(exps)
    return adv_image, norm, pred, probs, outputs


def apply_prog_run_model(program, prms, input=None):
    if input is None:
        input = program.source_image
    adv_image = apply_prog(program, input)

    # imshow_wrp(program.source_image, prms.dataset)
    # imshow_wrp(adv_image, prms.dataset)

    norm = torch.norm(torch.flatten(adv_image - input, start_dim=1), p=2, dim=1)
    outputs = prms.model(adv_image)
    pred = torch.max(outputs[0], 0)[1]
    exps = torch.exp(outputs[0])
    probs = exps / torch.sum(exps)
    return adv_image, norm, pred, probs, outputs


def generate_random_program(num_inst, x_image, true_label, target_label, patch_init_cfg, prms, device, s_bounds,
                            e_bounds):
    program = Program(true_label, target_label, x_image, prms.num_classes, device, s_bounds, e_bounds)
    for _ in range(num_inst):
        inst = PatchInstruction.generate(patch_init_cfg, prms, device, s_bounds, e_bounds)
        program.add_instruction(inst)
    return program


def get_loss(target_label, target_one_hot, outputs, norm, prms, program, adv_image):
    assert (outputs.max(1)[0] >= -prms.inf).all()
    # calculate loss
    max_others = torch.max((torch.ones_like(target_one_hot) - target_one_hot) * outputs - target_one_hot * prms.inf,
                           1)[0]
    loss_adv = torch.clamp(max_others - outputs[:, target_label], min=0.0)

    return loss_adv + prms.score_norm_reg * norm


def get_loss_gain(curr_loss, program, prms):
    adv_image, norm, _, _, outputs = apply_prog_run_model(program, prms)
    new_loss = get_loss(program.target_label, program.target_one_hot, outputs, norm, prms, program, adv_image)
    return curr_loss - new_loss


def get_start_ind_grad(program, inst_idx, curr_loss, prms, dims=(1,2)):
    inst = program.instructions[inst_idx]
    base_patch_start_ind = [inst.patch.start_ind[0].item(), inst.patch.start_ind[1].item(),
                            inst.patch.start_ind[2].item()]
    grad = [0,0,0]
    for dim in dims:
        inst.patch.start_ind[dim][0] = base_patch_start_ind[dim] + 1
        if inst.patch.is_legit():
            inst.patch.update_masks()
            loss_gain_pos = get_loss_gain(curr_loss, program, prms)
        else:
            loss_gain_pos = -prms.inf
        inst.patch.start_ind[dim][0] = base_patch_start_ind[dim] - 1
        if inst.patch.is_legit():
            inst.patch.update_masks()
            loss_gain_neg = get_loss_gain(curr_loss, program, prms)
        else:
            loss_gain_neg = -prms.inf

        inst.patch.start_ind[dim][0] = base_patch_start_ind[dim]
        inst.patch.update_masks()

        if loss_gain_pos > 0 and loss_gain_pos >= loss_gain_neg:
            grad[dim] = -loss_gain_pos.item()
        elif loss_gain_neg > 0 and loss_gain_neg > loss_gain_pos:
            grad[dim] = loss_gain_neg.item()
    return grad


def get_patch_shape_grad(program, inst_idx, curr_loss, prms, dims=(1,2)):
    inst = program.instructions[inst_idx]
    base_patch_shape = [inst.patch.patch_shape[0].item(), inst.patch.patch_shape[1].item(),
                        inst.patch.patch_shape[2].item()]
    grad = [0,0,0]
    for dim in dims:
        inst.patch.patch_shape[dim][0] = base_patch_shape[dim] + 1
        if inst.patch.is_legit():
            inst.patch.update_masks()
            loss_gain_pos = get_loss_gain(curr_loss, program, prms)
        else:
            loss_gain_pos = -prms.inf
        inst.patch.patch_shape[dim][0] = base_patch_shape[dim] - 1
        if inst.patch.is_legit():
            inst.patch.update_masks()
            loss_gain_neg = get_loss_gain(curr_loss, program, prms)
        else:
            loss_gain_neg = -prms.inf

        inst.patch.patch_shape[dim][0] = base_patch_shape[dim]
        inst.patch.update_masks()

        if loss_gain_pos > 0 and loss_gain_pos >= loss_gain_neg:
            grad[dim] = -loss_gain_pos.item()
        elif loss_gain_neg > 0 and loss_gain_neg > loss_gain_pos:
            grad[dim] = loss_gain_neg.item()
    return grad


def get_discrete_grads(program, optimizer, prms, params_to_update=None):
    grads = {}
    # grads for all discrete parameters
    if params_to_update is None:
        grads = get_all_discrete_grads(program, prms)
    else:
        assert len(params_to_update) == 1  # currently only supported in single-param mode
        for p in params_to_update:
            # parse parameter name
            inst_idx, param_type, dim_idx = optimizer.parse_param_name(p)
            # calculate base stats
            adv_image, norm, _, _, outputs = apply_prog_run_model(program, prms)
            curr_loss = get_loss(program.target_label, program.target_one_hot, outputs, norm, prms, program, adv_image)
            # call relevant function with dim
            if param_type == "start_ind":
                grads[p] = get_start_ind_grad(program, inst_idx, curr_loss, prms, dims=[dim_idx])[dim_idx]
            elif param_type == "patch_shape":
                grads[p] = get_patch_shape_grad(program, inst_idx, curr_loss, prms, dims=[dim_idx])[dim_idx]
    return grads


def discrete_optimization(program, optimizer, prms, mode):
    if mode == "GD":
        grads = get_discrete_grads(program, optimizer, prms)
        optimizer.step(program, grads)
    elif mode == "HC":
        for p in optimizer.params:
            params_to_update = [p]
            grads = get_discrete_grads(program, optimizer, prms, params_to_update)
            optimizer.step(program, grads, params_to_update)


def get_discrete_params(program, dims=(1,2)):
    params = {}
    for i, inst in enumerate(program.instructions):
        for d in dims:
            param_name = "inst" + str(i) + "_start_ind" + str(d)
            params[param_name] = inst.patch.start_ind[d]
        for d in dims:
            param_name = "inst" + str(i) + "_patch_shape" + str(d)
            params[param_name] = inst.patch.patch_shape[d]
    return params


def get_all_discrete_grads(program, prms):
    # run program to get current loss
    adv_image, norm, _, _, outputs = apply_prog_run_model(program, prms)
    curr_loss = get_loss(program.target_label, program.target_one_hot, outputs, norm, prms, program, adv_image)
    discrete_grads = {}
    for i, inst in enumerate(program.instructions):
        # get start_ind gradient
        start_ind_grad = get_start_ind_grad(program, i, curr_loss, prms)
        discrete_grads["inst" + str(i) + "_start_ind1"] = start_ind_grad[1]
        discrete_grads["inst" + str(i) + "_start_ind2"] = start_ind_grad[2]
        # get patch_shape gradient
        patch_shape_grad = get_patch_shape_grad(program, i, curr_loss, prms)
        discrete_grads["inst" + str(i) + "_patch_shape1"] = patch_shape_grad[1]
        discrete_grads["inst" + str(i) + "_patch_shape2"] = patch_shape_grad[2]

    return discrete_grads


def get_update_cntous_params(optimizer, program, prms):
    # print("before", str(program))
    adv_image, norm, _, _, outputs = apply_prog_run_model(program, prms)
    # calculate loss
    loss = get_loss(program.target_label, program.target_one_hot, outputs, norm, prms, program, adv_image)
    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if torch.count_nonzero(torch.isnan(program.instructions[0].scale)) != 0 or \
            torch.count_nonzero(torch.isnan(program.instructions[0].shift)) != 0:
        print("grads", program.instructions[0].scale.grad, program.instructions[0].shift.grad)
        print("after", str(program))
        exit(0)

    # grads = []
    # for p in optimizer.param_groups[0]['params']:
    #     grads.append(p.grad.item())
    # print("cntous grads", grads)


def set_best_patch_location(program, inst, stride, prms, s_bounds=None, e_bounds=None):
    assert inst in program.instructions

    if s_bounds is None:
        s_bounds = [0,0,0]
    if e_bounds is None:
        e_bounds = prms.image_shape

    best_loss = prms.inf
    best_start_ind = None
    for j in range(s_bounds[1], e_bounds[1] - int(inst.patch.patch_shape[1].item()) + 1, stride):
        for k in range(s_bounds[2], e_bounds[2] - int(inst.patch.patch_shape[2].item()) + 1, stride):
            inst.patch.start_ind[1][0] = j
            inst.patch.start_ind[2][0] = k
            inst.patch.update_masks()
            adv_image, norm, _, _, outputs = apply_prog_run_model(program, prms)
            loss = get_loss(program.target_label, program.target_one_hot, outputs, norm, prms, program, adv_image)
            if loss < best_loss:
                best_loss = loss.item()
                best_start_ind = [inst.patch.start_ind[0].item(), inst.patch.start_ind[1].item(),
                                  inst.patch.start_ind[2].item()]
    assert best_start_ind is not None
    inst.patch.start_ind[0][0] = best_start_ind[0]
    inst.patch.start_ind[1][0] = best_start_ind[1]
    inst.patch.start_ind[2][0] = best_start_ind[2]
    inst.patch.update_masks()


def gen_adv(x_image, true_label, target_label, num_instructions, optim_itr, score_norm_reg, num_candidates,
            loss_plateau_thresh, loss_plateau_patience, cntous_lr, dscrt_lr, stride, betas, cbetas, patch_init_cfg,
            s_bounds, e_bounds, out_dir, device, prms, queue):
    prms.model = prms.model.to(device)
    prms.score_norm_reg = score_norm_reg

    execute_cw = False
    if execute_cw:
        # generate CW attack
        cwattacker = advertorch.attacks.CarliniWagnerL2Attack(prms.model, num_classes=prms.num_classes, targeted=True)
        cw_start = time.time()
        cw_attack = cwattacker.perturb(x_image, y=torch.tensor([target_label.item()], device=device))
        cw_time = int((time.time() - cw_start) / 60)
        cw_pert = cw_attack - x_image
        # CW attack stats
        cw_outputs = prms.model(cw_attack)
        cw_exps = torch.exp(cw_outputs[0])
        cw_probs = cw_exps / torch.sum(cw_exps)
        cw_pred = torch.max(cw_outputs[0], 0)[1]
        cw_success = cw_pred == target_label
        cw_l0_norm = torch.norm(torch.flatten(cw_attack - x_image, start_dim=1), p=0, dim=1)
        cw_l1_norm = torch.norm(torch.flatten(cw_attack - x_image, start_dim=1), p=1, dim=1)
        cw_l2_norm = torch.norm(torch.flatten(cw_attack - x_image, start_dim=1), p=2, dim=1)
        cw_linf_norm = torch.norm(torch.flatten(cw_attack - x_image, start_dim=1), p=float('inf'), dim=1)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'log.txt')
    log_file_lock_file = log_file + ".lock"
    open(log_file_lock_file, 'a').close()
    log_file_lock = FileLock(log_file_lock_file)

    # calc base probs
    outputs = prms.model(x_image)
    exps = torch.exp(outputs[0])
    orig_probs = exps / torch.sum(exps)
    pred = torch.max(outputs[0], 0)[1]
    assert pred.item() == true_label.item()

    loss_values, norm_values, success_values = [], [], []
    inst_itr_markers = []

    success = False
    best_loss = prms.inf
    tot_itr = 0
    best_itr = 0
    start = time.time()
    orig_instructions = []

    program = Program(true_label, target_label, x_image, prms.num_classes, device, s_bounds, e_bounds)
    for n in range(num_instructions):
        orig_instructions_per_itr = []
        inst_itr_markers.append(tot_itr)
        best_cnd = None
        itr_per_cnd = [None] * num_candidates
        loss_vals_per_cnd = [[] for _ in range(num_candidates)]
        norm_vals_per_cnd = [[] for _ in range(num_candidates)]
        success_vals_per_cnd = [[] for _ in range(num_candidates)]
        for c in range(num_candidates):
            cnd_program = Program.copy(program)
            inst = PatchInstruction.generate(patch_init_cfg, prms, device, s_bounds, e_bounds)
            cnd_program.add_instruction(inst)

            # init instruction patch to best location
            set_best_patch_location(cnd_program, inst, stride, prms, s_bounds, e_bounds)

            orig_instructions_per_itr.append(PatchInstruction(Patch([t.detach().clone() for t in inst.patch.start_ind],
                                                    [t.detach().clone() for t in inst.patch.patch_shape],
                                                    inst.patch.image_shape, inst.device, s_bounds, e_bounds),
                                                    inst.scale, inst.shift, inst.device))

            # optimizer = optim.Adam([inst.shift for inst in cnd_program.instructions], lr=cntous_lr, betas=cbetas)
            optimizer = optim.Adam([inst.scale for inst in cnd_program.instructions] +
                                   [inst.shift for inst in cnd_program.instructions], lr=cntous_lr, betas=cbetas)

            dscrte_params = get_discrete_params(cnd_program)
            dscrte_optimizer = DiscreteAdam(dscrte_params, lr=dscrt_lr, betas=betas)

            plteu_itr = 0
            loss_anchor = prms.inf
            ii = 0
            while ii < optim_itr and plteu_itr < loss_plateau_patience:

                discrete_optimization(cnd_program, dscrte_optimizer, prms, mode="HC")
                get_update_cntous_params(optimizer, cnd_program, prms)

                adv_image, norm, pred, _, outputs = apply_prog_run_model(cnd_program, prms)
                loss = get_loss(cnd_program.target_label, cnd_program.target_one_hot, outputs, norm, prms, cnd_program,
                                adv_image)
                if pred == target_label:
                    success = True
                if loss < best_loss and (pred == target_label or not success):
                    best_loss = loss.item()
                    best_program_compact = cnd_program.save_compact()
                    best_itr = tot_itr + ii
                    best_cnd = c
                # count number of iterations with no significant changes
                if loss <= loss_anchor * (1-loss_plateau_thresh):
                    loss_anchor = loss.item()
                    plteu_itr = 0
                else:
                    plteu_itr += 1
                ii += 1

                loss_vals_per_cnd[c].append(loss.item())
                norm_vals_per_cnd[c].append(norm.item())
                success_vals_per_cnd[c].append(int(pred.item() == target_label))
            itr_per_cnd[c] = ii

        if best_cnd is not None:
            program = Program.restore_from_compact(best_program_compact, true_label, target_label,
                                                   x_image.detach().clone(), prms.num_classes, device, s_bounds,
                                                   e_bounds)
            tot_itr += itr_per_cnd[best_cnd]
            loss_values.extend(loss_vals_per_cnd[best_cnd])
            norm_values.extend(norm_vals_per_cnd[best_cnd])
            success_values.extend(success_vals_per_cnd[best_cnd])
            orig_instructions.append(orig_instructions_per_itr[best_cnd])

    ######################################################################################################

    end = time.time()
    attack_time = int((end - start) / 60)
    # save best program
    best_program = Program.restore_from_compact(best_program_compact, true_label, target_label,
                                                x_image.detach().clone(), prms.num_classes, device, s_bounds, e_bounds)
    adv_image, norm, pred, probs, _ = apply_prog_run_model(best_program, prms)
    l0_norm = torch.norm(torch.flatten(adv_image - best_program.source_image, start_dim=1), p=0, dim=1)
    l1_norm = torch.norm(torch.flatten(adv_image - best_program.source_image, start_dim=1), p=1, dim=1)
    linf_norm = torch.norm(torch.flatten(adv_image - best_program.source_image, start_dim=1), p=float('inf'), dim=1)
    success = pred == target_label
    if success:
        print("%d=>%d - success, best_itr = %d, tot_itr = %d" % (true_label, target_label, best_itr, tot_itr))
        dir_path = get_dir_path(out_dir, true_label.item(), target_label.item(), norm.item(), prms)
        queue.put(norm.item())
        with log_file_lock:
            open(log_file, 'a').write("%d=>%d. attack generation took %d minutes, norm = %f (%s)" % (
                true_label, target_label, attack_time, norm.item(), device) + "\n")
    else:
        print("%d=>%d - failure, best_itr = %d, tot_itr = %d" % (true_label, target_label, best_itr, tot_itr))
        prob_diff = probs[pred].item() - probs[target_label].item()
        dir_path = get_dir_path(out_dir, true_label.item(), target_label.item(), norm.item(), prms, failed=True,
                                prob_diff=prob_diff)

    with open(os.path.join(dir_path, 'program.txt'), 'w') as f:
        # print(str(best_program), file=f)
        print(print_program_with_orig(best_program, orig_instructions), file=f)
    torch.save(best_program, os.path.join(dir_path, "program.pt"))

    if execute_cw:
        vis.plot_attack_cw(x_image, adv_image, cw_attack, cw_success.item(), cw_l2_norm.item(), cw_l0_norm.item(),
                    cw_l1_norm.item(), cw_linf_norm.item(), true_label.item(), target_label.item(), success.item(),
                    norm.item(), l0_norm.item(), l1_norm.item(), linf_norm.item(), probs, orig_probs, prms,
                    savedir=dir_path, instructions=best_program.instructions)
    else:
        vis.plot_attack(x_image, adv_image, true_label.item(), target_label.item(), success.item(),
                    norm.item(), l0_norm.item(), l1_norm.item(), linf_norm.item(), probs, orig_probs, prms,
                    savedir=dir_path, instructions=best_program.instructions)
    vis.plot_loss_norm_success(tot_itr, loss_values, norm_values, success_values, best_itr, inst_itr_markers,
                               savedir=dir_path)

    d = {'true_label': true_label.item(), 'target_label': target_label.item(),
         'num_instructions': len(best_program.instructions), 'success': success.item(), 'l0_norm': int(l0_norm.item()),
         'l1_norm': round(l1_norm.item(), 2), 'l2_norm': round(norm.item(), 2), 'linf_norm': round(linf_norm.item(), 2),
         'attack_time': attack_time, 'parallel': prms.parallel}
    torch.save(d, os.path.join(dir_path, "program_stats.pt"))
    with open(os.path.join(dir_path, 'program_stats.txt'), 'w') as f:
        print(str(d), file=f)

    if execute_cw:
        torch.save(cw_pert, os.path.join(dir_path, "cw_pert.pt"))
        cw_d = {'true_label': true_label.item(), 'target_label': target_label.item(), 'success': cw_success.item(),
             'l0_norm': int(cw_l0_norm.item()), 'l1_norm': round(cw_l1_norm.item(), 2),
             'l2_norm': round(cw_l2_norm.item(), 2), 'linf_norm': round(cw_linf_norm.item(), 2),
             'attack_time': cw_time, 'parallel': 1}
        torch.save(cw_d, os.path.join(dir_path, "cw_stats.pt"))
        with open(os.path.join(dir_path, 'cw_stats.txt'), 'w') as f:
            print(str(cw_d), file=f)

