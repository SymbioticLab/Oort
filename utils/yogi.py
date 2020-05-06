import torch

class YoGi():
    def __init__(self, eta=1e-2, tau=1e-3, beta=0.999):
        self.eta = eta
        self.tau = tau
        self.beta = beta

        self.v_t = []

    def update(self, gradients):
        update_gradients = []
        for idx, gradient in enumerate(gradients):
            gradient_square = gradient * gradient
            if len(self.v_t) <= idx:
                self.v_t.append(gradient_square)
            else:
                self.v_t[idx] = self.v_t[idx] - (1.-self.beta) * gradient_square * torch.sign(self.v_t[idx] - gradient_square)
                update_gradients.append(self.eta * (gradient/(torch.sqrt(self.v_t[idx]) + self.tau)))

        if len(update_gradients) == 0:
            update_gradients = gradients

        return update_gradients
