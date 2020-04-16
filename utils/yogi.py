import torch

class YoGi(nn.Module):
    def __init__(self, eta=5e-2, tau=1e-3, beta=0.999):
        self.eta = eta
        self.tau = tau
        self.beta = beta

        self.v_t = []

    def update(gradients):
        update_gradients = []
        for idx, gradient in enumerate(gradients):
            if len(self.v_t) <= idx:
                self.v_t.append(gradient)

            gradient_square = gradient ** 2
            self.v_t[idx] = self.v_t[idx] - (1.-self.beta) * gradient_square * torch.sign(self.v_t[idx] - gradient_square)

            update_gradients.append(gradient + self.eta * (gradient/(torch.sqrt(self.v_t[idx]) + self.tau)))

        return update_gradients
