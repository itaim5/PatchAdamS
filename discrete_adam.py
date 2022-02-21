import math
import re

class DiscreteAdam():

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.wd = weight_decay

        self.params = params
        self.states = {}
        for p in self.params:
            self.states[p] = {}

    def step(self, program, grads, params_to_update=None):
        if params_to_update is None:
            params_to_update = self.params
        assert len(grads) == len(params_to_update)
        for p in params_to_update:
            state = self.states[p]
            # Lazy state initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = 0
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = 0

            # update the steps for each param group update
            state['step'] += 1

            # bias correction updates
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']

            # add weight decay
            grads[p] += self.params[p].item() * self.wd

            # Decay the first and second moment running average coefficient
            state['exp_avg'] = state['exp_avg'] * self.beta1 + grads[p] * (1-self.beta1)
            state['exp_avg_sq'] = state['exp_avg_sq'] * self.beta2 + (grads[p] ** 2) * (1-self.beta2)

            step_size = self.lr / bias_correction1
            denom = math.sqrt(state['exp_avg_sq']) / math.sqrt(bias_correction2) + self.eps

            self.params[p] -= step_size * state['exp_avg'] / denom

            # make sure change is valid
            inst_idx, _, _ = self.parse_param_name(p)
            patch = program.instructions[inst_idx].patch
            if patch.is_legit():
                patch.update_masks()
            else:
                self.params[p] += step_size * state['exp_avg'] / denom

    @staticmethod
    def parse_param_name(param_name):
        inst_idx = int(re.match("inst(\d+).*", param_name).group(1))
        param_type = re.match("inst\d+_(.*)\d", param_name).group(1)
        dim_idx = int(re.match("inst\d+_.*(\d)", param_name).group(1))
        return inst_idx, param_type, dim_idx