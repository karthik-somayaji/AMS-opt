import sys
sys.path.append("../")

from utils.dataset import get_subset, get_topk_examples # noqa: E402


class Sampler(object):
    def __init__(self, n_sample, method:str='topk', shuffle=False):
        """ a class for sampling data from the collected data

        Args:
            n_topk (int, optional): the number of top-k samples. Defaults to 5.
            n_random (int, optional): the number of random samples. Defaults to 5.
            method: should be ["topk", "random"]
            shuffle: whether to shuffle the samples;
        """
        self.n_sample = n_sample
        self.method = method
        self.shuffle = shuffle

    def sample(self, data_collected):
        n_topk = 0
        n_random = 0
        n_data = len(data_collected["targets"])
        n_max = min(self.n_sample, n_data)
        if self.method == "topk":
            n_topk = n_max
        elif self.method == "random":
            n_random = n_max
        elif self.method == "all":
            return data_collected
        elif self.method == "mixed":
            n_topk = n_max // 2
            n_random = n_max - n_topk
        #return get_subset(data_collected, n_topk, n_random)
        return get_topk_examples(data_collected, n_topk)


if __name__ == "__main__":
    pass
