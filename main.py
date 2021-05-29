import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from config import Config

from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.softmax_classifier import Softmax_Layer
from model.memory_network.attention_memory import Attention_Memory
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified

from utils import outputer

from sampler import data_sampler
from data_loader import get_data_loader

def transfer_to_device(list_ins, device):
    import torch
    for ele in list_ins:
        if isinstance(ele,list):
            for x in ele:
                x.to(device)
        if isinstance(ele,torch.Tensor):
            ele.to(device)
    return list_ins

# Done
def get_proto(config, encoder, mem_set):
    # aggregate the prototype set for further use.
    data_loader = get_data_loader(config, mem_set, False, False, 1)

    features = []
    for step, (labels, tokens) in enumerate(data_loader):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        features.append(feature)
    features = torch.cat(features, dim=0)

    proto = torch.mean(features, dim=0, keepdim=True)
    
    # return the averaged prototype
    return proto

# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(config, encoder, sample_set):
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []

    for step, (labels, tokens) in enumerate(data_loader):
        tokens=torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = encoder(tokens).cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    mem_set = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = sample_set[sel_index]
        mem_set.append(instance)
    return mem_set

def train_simple_model(config, encoder, classifier, training_data, epochs):

    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': encoder.parameters(), 'lr': 0.00001},
                            {'params': classifier.parameters(), 'lr': 0.001}
                            ])

    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens) in enumerate(data_loader):
            encoder.zero_grad()
            classifier.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            logits = classifier(reps)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")

def train_mem_model(config, encoder, classifier, memory_network, training_data, mem_data, epochs):
    data_loader = get_data_loader(config, training_data)
    encoder.train()
    classifier.train()
    memory_network.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001},
        {'params': memory_network.parameters(), 'lr': 0.0001}
    ])

    # mem_data.unsqueeze(0)
    # mem_data = mem_data.expand(data_loader.batch_size, -1, -1)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens) in enumerate(data_loader):

            mem_for_batch = mem_data.clone()
            mem_for_batch.unsqueeze(0)
            mem_for_batch = mem_for_batch.expand(len(tokens), -1, -1)

            encoder.zero_grad()
            classifier.zero_grad()
            memory_network.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            reps = memory_network(reps, mem_for_batch)
            logits = classifier(reps)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(memory_network.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")

def evaluate_model(config, encoder, classifier, memory_network, test_data, protos4eval):
	data_loader = get_data_loader(config, test_data, batch_size=1)
	encoder.eval()
	classifier.eval()
	memory_network.eval()
	n = len(test_data)

	correct = 0
	protos4eval.unsqueeze(0)
	protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
	for step, (labels, tokens) in enumerate(data_loader):
		mem_for_batch = protos4eval.clone()
		labels = labels.to(config.device)
		tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
		reps = encoder(tokens)
		reps = memory_network(reps, mem_for_batch)
		logits = classifier(reps)

		neg_index = random.sample(range(0, 80), 10)
		neg_sim = logits[:,neg_index].cpu().data.numpy()
		max_smi = np.max(neg_sim,axis=1)

		label_smi = logits[:,labels].cpu().data.numpy()

		if label_smi >= max_smi:
		    correct += 1

	return correct/n

def evaluate_no_mem_model(config, encoder, classifier, test_data):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, (labels, tokens) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        neg_index = random.sample(range(0, 80), 10)
        neg_sim = logits[:,neg_index].cpu().data.numpy()
        max_smi = np.max(neg_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n

def evaluate_strict_model(config, encoder, classifier, memory_network, test_data, protos4eval,seen_relations):
	data_loader = get_data_loader(config, test_data, batch_size=1)
	encoder.eval()
	classifier.eval()
	memory_network.eval()
	n = len(test_data)

	correct = 0
	protos4eval.unsqueeze(0)
	protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
	for step, (labels, tokens) in enumerate(data_loader):
		mem_for_batch = protos4eval.clone()
		labels = labels.to(config.device)
		tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
		reps = encoder(tokens)
		reps = memory_network(reps, mem_for_batch)
		logits = classifier(reps)

		seen_relation_ids = [rel2id[relation] for relation in seen_relations]
		seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
		max_smi = np.max(seen_sim,axis=1)

		label_smi = logits[:,labels].cpu().data.numpy()

		if label_smi >= max_smi:
		    correct += 1

	return correct/n

def evaluate_strict_no_mem_model(config, encoder, classifier, test_data, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, (labels, tokens) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n

if __name__ == '__main__':

    parser = ArgumentParser(
        description="Config for lifelong relation extraction (classification)")
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    # output result
    printer = outputer()
    middle_printer = outputer()
    start_printer=outputer()

    # set training batch
    for i in range(config.total_round):

        test_cur = []
        test_total = []

        # set random seed
        random.seed(config.seed+i*100)

        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed+i*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        # encoder setup
        encoder = Bert_Encoder(config=config).to(config.device)
        # classifier setup
        classifier = Softmax_Layer(input_size=encoder.output_size, num_class=config.num_of_relation).to(config.device)

        # record testing results
        sequence_results = []
        result_whole_test = []

        # initialize memory and prototypes
        num_class = len(sampler.id2rel)
        memorized_samples = {}

        # load data and start computation
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):

            print(current_relations)

            temp_mem = {}
            temp_protos = []
            for relation in seen_relations:
                if relation not in current_relations:
                    temp_protos.append(get_proto(config, encoder, memorized_samples[relation]))
                    
            # Initial
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]
            # train model
            train_simple_model(config, encoder, classifier, train_data_for_initial, config.step1_epochs)


            # # Memory Activation
            # train_data_for_replay = []
            # random.seed(config.seed+i*100)
            # for relation in current_relations:
            #     train_data_for_replay += training_data[relation]
            # for relation in memorized_samples:
            #     train_data_for_replay += memorized_samples[relation]
            # train_simple_model(config, encoder, classifier, train_data_for_replay, config.step2_epochs)

            for relation in current_relations:
                temp_mem[relation] = select_data(config, encoder, training_data[relation])
                temp_protos.append(get_proto(config, encoder, temp_mem[relation]))
            temp_protos = torch.cat(temp_protos, dim=0).detach()


            memory_network = Attention_Memory_Simplified(mem_slots=len(seen_relations),
                                              input_size=encoder.output_size,
                                              output_size=encoder.output_size,
                                              key_size=config.key_size,
                                              head_size=config.head_size
                                              ).to(config.device)

            # generate training data for the corresponding memory model (ungrouped)
            train_data_for_memory = []
            for relation in temp_mem.keys():
                train_data_for_memory += temp_mem[relation]
            for relation in memorized_samples.keys():
                train_data_for_memory += memorized_samples[relation]
            random.shuffle(train_data_for_memory)
            train_mem_model(config, encoder, classifier, memory_network, train_data_for_memory, temp_protos, config.step3_epochs)

            # regenerate memory
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])
            protos4eval = []
            for relation in memorized_samples:
                protos4eval.append(get_proto(config, encoder, memorized_samples[relation]))
            protos4eval = torch.cat(protos4eval, dim=0).detach()

            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]

            cur_acc = evaluate_strict_model(config, encoder, classifier, memory_network, test_data_1, protos4eval,seen_relations)
            total_acc = evaluate_strict_model(config, encoder, classifier, memory_network, test_data_2, protos4eval,seen_relations)
            # cur_acc = evaluate_strict_no_mem_model(config, encoder, classifier, test_data_1, seen_relations)
            # total_acc = evaluate_strict_no_mem_model(config, encoder, classifier, test_data_2, seen_relations)

            # encoder.save_parameters('./model_parameters/abalation_study/FewRel_10tasks_encoder_task' + str(steps+1)+'.json')
            # classifier.save_parameters('./model_parameters/abalation_study/FewRel_10tasks_classifier_task' + str(steps+1)+'.json')
            # memory_network.save_parameters('./model_parameters/abalation_study/FewRel_10tasks_memory_network_task' + str(steps+1)+'.json')
            # np.save('./model_parameters/abalation_study/FewRel_10tasks_mem_task' + str(steps+1)+'.npy', protos4eval.cpu().numpy())

            print(f'Restart Num {i+1}')
            print(f'task--{steps + 1}:')
            print(f'current test acc:{cur_acc}')
            print(f'history test acc:{total_acc}')
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            print(test_cur)
            print(test_total)

