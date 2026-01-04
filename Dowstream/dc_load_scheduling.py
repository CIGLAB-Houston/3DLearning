import torch


class DC_Scheduling():
    def __init__(self, eta, a, b, capability_range, power_range):
        self.LLM_CAPABILITY_RANGE = capability_range[1:]
        self.LLM_CAPABILITY_MIN = capability_range[1]
        self.LLM_CAPABILITY_MAX = capability_range[2]


        self.ETA = eta
        self.A = a
        self.B = b

        self.LLM_POWER_RANGE = power_range[1:]
        self.LLM_IDLE_POWER = power_range[0]
        self.LLM_ACTIVATE_POWER_MAX = power_range[1]
        self.LLM_ACTIVATE_POWER_MIN = power_range[2]


    def acc_curve(self, x):
        s = torch.log(self.A * x + 1) / (self.B * torch.log(torch.tensor(self.A + 1.0)))
        return s

    def utility(self, x, y):
        y = torch.clamp(y, min=1e-8) ##
        z = torch.where(x >= y, torch.ones_like(x), x / y)
        s = self.acc_curve(z)
        return s * y

    def cost(self, x, y):
        enable_process_tokens = torch.minimum(x, y)
        relu = torch.relu(x - y)
        llm_activate_per_token = self.LLM_ACTIVATE_POWER_MAX / (self.LLM_CAPABILITY_MIN)
        llm_idle_per_token = self.LLM_IDLE_POWER / (self.LLM_CAPABILITY_MIN)
        cost = llm_activate_per_token * enable_process_tokens + llm_idle_per_token * relu
        return cost

    def loss(self, x, y):
        norm_x = self.norm_capability(z=x)
        norm_y = self.norm_capability(z=y)
        first = (1 - self.ETA) * self.cost(x=norm_x, y=norm_y)
        second = self.ETA * self.utility(x=norm_x, y=norm_y)
        L = first - second
        return L

    def norm_capability(self, z):
        delta = self.LLM_CAPABILITY_MAX - self.LLM_CAPABILITY_MIN
        z_norm = torch.relu((z - self.LLM_CAPABILITY_MIN) / delta)
        return z_norm



    def find_best_scheduling(self, y):
        x_min = self.LLM_CAPABILITY_MIN
        x_max = self.LLM_CAPABILITY_MAX
        llm_activate_per_token = self.LLM_ACTIVATE_POWER_MAX / self.LLM_CAPABILITY_MIN
        best_x = (y * self.ETA) / ((1 - self.ETA) * llm_activate_per_token * self.B * torch.log(
            torch.tensor(self.A + 1.0))) - y / self.A

        best_x_clamped_1 = torch.clamp(best_x, x_min, x_max)
        lowest_loss_1 = self.loss(best_x_clamped_1, y)

        best_x_clamped_2 = torch.clamp(y, x_min, x_max)
        lowest_loss_2 = self.loss(best_x_clamped_2, y)

        if torch.mean(lowest_loss_1) <= torch.mean(lowest_loss_2):
            lowest_loss = lowest_loss_1
            best_x_clamped = best_x_clamped_1
        else:
            lowest_loss = lowest_loss_2
            best_x_clamped = best_x_clamped_1

        if torch.isnan(best_x_clamped).any() or torch.isinf(best_x_clamped).any():
            print("[Warning] find_best_scheduling produced nan or inf")
        return best_x_clamped, lowest_loss
