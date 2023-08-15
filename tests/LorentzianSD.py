import numpy as np


class LorEnv(object):
    def __init__(self, eta, p, gamma, wc):
        """
        defines a SD as J(w) = eta * sum_i p_i * gamma_i / ( gamma_i^2 + (wc_i-w)^2 )
        """
        self.eta = eta
        self.p = np.asarray(p)
        self.gamma = np.asarray(gamma)
        self.wc = np.asarray(wc)
        self.p_gamma = self.p * self.gamma
        self.gamma_sq = self.gamma**2

    def __j(self, wi):
        return self.eta * np.sum(self.p_gamma / (self.gamma_sq + (self.wc - wi) ** 2))

    def __s(self, wi):
        return -self.eta * np.sum(
            self.p * (self.wc - wi) / (self.gamma_sq + (self.wc - wi) ** 2)
        )

    def __bcf(self, taui):
        return self.eta * np.sum(
            self.p * np.exp(-self.gamma * np.abs(taui)) * np.exp(-1j * self.wc * taui)
        )

    def J(self, w):
        try:
            return np.asarray([self.__j(wi) for wi in w])
        except:
            return self.__j(w)

    def S(self, w):
        try:
            return np.asarray([self.__s(wi) for wi in w])
        except:
            return self.__s(w)

    def F(self, w):
        return self.J(w) + 1j * self.S(w)

    def bcf(self, tau):
        try:
            return np.asarray([self.__bcf(taui) for taui in tau])
        except:
            return self.__bcf(tau)

    def get_BCF_class(self):
        return LorBCF(self.eta, self.p, self.gamma, self.wc)

    def get_SD_class(self):
        return LorSD(self.eta, self.p, self.gamma, self.wc)

    def get_S_class(self):
        return LorSD_S(self.eta, self.p, self.gamma, self.wc)

    def __bfkey__(self):
        return [self.eta, self.p, self.gamma, self.wc]


class LorBCF(object):
    def __init__(self, eta, p, gamma, wc):
        self.lorEnv = LorEnv(eta=eta, p=p, gamma=gamma, wc=wc)

    def __call__(self, tau):
        return self.lorEnv.bcf(tau)

    def __bfkey__(self):
        return self.lorEnv.__bfkey__()


class LorSD(object):
    def __init__(self, eta, p, gamma, wc):
        self.lorEnv = LorEnv(eta=eta, p=p, gamma=gamma, wc=wc)

    def __call__(self, w):
        return self.lorEnv.J(w)

    def __bfkey__(self):
        return self.lorEnv.__bfkey__()


class LorSD_S(object):
    def __init__(self, eta, p, gamma, wc):
        self.lorEnv = LorEnv(eta=eta, p=p, gamma=gamma, wc=wc)

    def __call__(self, w):
        return self.lorEnv.S(w)

    def __bfkey__(self):
        return self.lorEnv.__bfkey__()


def get_eta(S_0, gamma, wc):
    return S_0 * (gamma**2 + wc**2) / wc


def get_J_at_wc(S_0, gamma, wc):
    return get_eta(S_0, gamma, wc) / gamma
