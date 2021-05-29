import torch
import random
import numpy as np
import time
import sys

def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)

class outputer(object):

    def __init__(self):
        self.start_time = time.time()
        self.all_results = []
        self.result_all_test_data = []
        self.sequence_times = 0

    def init(self):
        self.start_time = time.time()
        self.all_results = []
        self.result_all_test_data = []
        self.sequence_times = 0

    def append(self, sequence_results = None, result_whole_test = None):
        if not isinstance(sequence_results, type(None) ):
            self.all_results.append(sequence_results)
        if not isinstance(result_whole_test, type(None)):
            self.result_all_test_data.append(result_whole_test)
        self.sequence_times += 1.0

    def print_avg_results(self, all_results):
        avg_result = []
        for i in range(len(all_results[0])):#代表了k shot
            avg_result.append(np.average([result[i] for result in all_results], 0))
        for line_result in avg_result:
            self.print_list(line_result)
        return avg_result

    def print_avg_cand(self, sample_list):
        cand_lengths = []
        for sample in sample_list:
            cand_lengths.append(len(sample[1]))
        print('avg cand size:', np.average(cand_lengths))

    def print_list(self, result):
        for num in result:
            sys.stdout.write('%.3f, ' %num)
        print('')

    def output(self):
        avg_result_all_test = np.average(self.result_all_test_data, 0)
        for result_whole_test in self.result_all_test_data:
            self.print_list(result_whole_test)
        print ("-------------------------------------------")
        self.print_list(avg_result_all_test)
        print ("===========================================")
        all_results = self.print_avg_results(self.all_results)
        # np.array(self.all_results)
        # print (all_results)
        print (all_results[-1].mean())