import argparse
import itertools
import json
import random
from copy import copy

import numpy as np


def init_config(config):
    """
    预处理配置文件，将属性映射到自然数，并检测属性是否一致
    :param config: 配置文件，dict格式
    :return:
    """
    attributes = set(config['domain'].keys())
    attributes_set = set(attributes)
    attr_map = {x: i for i, x in enumerate(attributes)}

    domain = dict()
    for attr, size in config['domain'].items():
        domain[attr_map[attr]] = size

    check = set()
    distribution = dict()
    for node, info in config['distribution'].items():
        assert set(info['from']) <= attributes_set
        assert node in attributes_set and node not in check
        assert len(info['max_dependency_internal']) in {0, 2}
        check.add(node)
        tmp = copy(info)
        tmp['from'] = list(map(lambda x: attr_map[x], info['from']))
        distribution[attr_map[node]] = tmp

    return domain, distribution, attributes


def topological_sequence(distribution, domain):
    """
    获得拓扑序
    :param distribution: 依赖关系
    :param domain: 值域
    :return:
    """
    in_degrees = np.zeros(len(domain))
    out_edge = dict()

    for key, info in distribution.items():
        in_degrees[key] += len(info['from'])
        for node in info['from']:
            out_edge.setdefault(node, [])
            out_edge[node].append(key)

    ret = []
    while len(ret) < len(domain):
        zero_degrees = np.where(in_degrees == 0)[0]
        # 防止出现环
        assert len(zero_degrees) > 0
        in_degrees[zero_degrees] = -1
        for node in zero_degrees:
            out_edge.setdefault(node, [])
            in_degrees[out_edge[node]] -= 1
        ret.extend(zero_degrees)
    return ret


def generate_data(domain, distribution, seq, number=10000):
    """
    生成数据
    :param domain: 值域
    :param distribution: 分布
    :param seq: 节点生成序列（拓扑序）
    :param number: 数据数量
    :return:
    """

    def split_internal(size, number):
        # 将区间[0,size)均匀的划分为number个区间
        assert number <= size
        unit_len = size / number
        ret = [(round(unit_len * i), round(unit_len * (i + 1))) for i in range(number)]
        return ret

    data = np.zeros((number, len(domain)))
    for attr in seq:
        method = 'uniform' if attr not in distribution else distribution[attr]['default']

        if attr in distribution and len(distribution[attr]['from']) > 0:
            internals = []
            for node in distribution[attr]['from']:
                candidates = np.arange(domain[node])
                weights = np.exp(-candidates)
                weights = weights / weights.sum()
                internals.append(split_internal(domain[node], np.random.choice(candidates, p=weights) + 1))

            for x in itertools.product(*internals):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    with open('config.json', 'r', encoding='utf-8') as df:
        config = json.load(df)

    domain, distribution, attributes_name = init_config(config)
    seq = topological_sequence(distribution, domain)
    generate_data(domain, distribution, seq)
