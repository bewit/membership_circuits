import torch
from typing import Dict, List, Optional, Union
import numpy as np
from copy import deepcopy
import pickle

from src.models.nodewise.nodes import Node, ProductNode, SumNode, LeafNode, get_nodes_by_type


def prune_pc(node: Node, contract_single_parents: bool = True):
    node.is_valid()
    nodes = get_nodes_by_type(node, (ProductNode, SumNode))

    while len(nodes) > 0:
        n = nodes.pop()

        n_type = type(n)
        is_sum = n_type == SumNode

        i = 0
        while i < len(n.children_circuit):
            c = n.children_circuit[i]

            # if my children has only one node, we can get rid of it and link directly to that grandchildren
            if contract_single_parents and not isinstance(c, LeafNode) and len(c.children_circuit) == 1:
                n.children_circuit[i] = c.children_circuit[0]
                continue

            if n_type == type(c):
                del n.children_circuit[i]
                n.children_circuit.extend(c.children_circuit)

                if is_sum:
                    weights = torch.exp(n.log_weights).cpu().detach().numpy().tolist()
                    w = weights[i]
                    del weights[i]

                    children_weights = torch.exp(c.log_weights).cpu().detach().numpy().tolist()
                    weights.extend([cw * w for cw in children_weights])
                    assert np.isclose(sum(weights), 1.0)
                    n.log_weights = torch.nn.Parameter(torch.log(torch.tensor(weights)))
                continue

            i += 1
        if is_sum and i > 0:
            weights = torch.exp(n.log_weights).cpu().detach().numpy().tolist()
            weights[0] = 1.0 - sum(weights[1:])
            n.log_weights = torch.nn.Parameter(torch.log(torch.tensor(weights)))

    if contract_single_parents and isinstance(node, (ProductNode, SumNode)) and len(node.children_circuit) == 1:
        node = node.children_circuit[0]

    node.is_valid()
    return node


def structural_marginalization(node: Node, keep_scope: list[int]):
    keep_scope = set(keep_scope)

    def marginalize_recursive(node):
        new_node_scope = keep_scope.intersection(set(node.scope))

        if len(new_node_scope) == 0:
            return None
        
        if isinstance(node, LeafNode):
            if len(node.scope) > 1:
                raise ValueError("Encountered leaf node with |scope| > 1")
            
            return pickle.loads(pickle.dumps(node))

        new_children = []
        for child in node.children_circuit:
            new_child = marginalize_recursive(child)
            if new_child is None:
                continue
            new_children.append(new_child)

        if isinstance(node, SumNode):
            node_log_weights = node.log_weights
            node_log_weights = torch.log_softmax(node_log_weights, -1)
            new_node = SumNode(scope=list(new_node_scope), children_circuit=new_children, log_weights=node_log_weights)
        elif isinstance(node, ProductNode):
            new_node = ProductNode(scope=list(new_node_scope), children_circuit=new_children)
        else:
            raise ValueError()
        
        return new_node
        
    new_node = marginalize_recursive(node)
    new_node = prune_pc(new_node)
    new_node.is_valid()
    
    return new_node



def structural_conditioning(node: Node, evidence: torch.Tensor, epsilon=1e-10):
    def sum_condition(node: SumNode, input_vals=None, scope=None):
        if not scope.intersection(node.scope):
            return pickle.loads(pickle.dumps(node)), 0
        new_scope = list(set(node.scope) - scope)
        new_children = []
        new_weights = []
        old_node_weights = node._get_linear_weights()
        probs = []
        for i, c in enumerate(node.children_circuit):
            result = recurse_pc(c, input_vals, scope)
            if result[0]:
                new_children.append(result[0])
                new_weights.append(old_node_weights[i] * torch.exp(result[1]).item())
            else:
                probs.append(old_node_weights[i].item() * torch.exp(result[1]).item())
        new_weights = torch.tensor(new_weights) + epsilon
        new_log_weights = torch.log(torch.tensor([w / torch.sum(new_weights) for w in new_weights]))
        if torch.any(torch.isnan(new_log_weights)):
            raise ValueError("Found nan weights")
        if not new_scope:
            return None, torch.log(torch.sum(torch.tensor(probs)))
        
        new_node = SumNode(scope=new_scope, children_circuit=new_children, log_weights=new_log_weights)
        return new_node, torch.log(torch.sum(torch.tensor(new_weights)))

    def prod_condition(node: ProductNode, input_vals=None, scope=None):
        if not scope.intersection(node.scope):
            return pickle.loads(pickle.dumps(node)), 0
        new_scope = list(set(node.scope) - scope)
        log_prob = 0.0

        new_children = []
        for c in node.children_circuit:
            result = recurse_pc(c, input_vals, scope)
            if result[0]:
                new_children.append(result[0])
            log_prob += float(result[1])
        new_node = ProductNode(scope=new_scope, children_circuit=new_children)
        return new_node, torch.tensor(log_prob)

    def leaf_condition(node: LeafNode, input_vals=None, scope=None):
        if not scope.intersection(node.scope):
            return pickle.loads(pickle.dumps(node)), 0
        
        ll = node.log_pdf(input_vals)
        return None, ll


    def recurse_pc(node, input_vals, scope):
        if isinstance(node, SumNode):
            return sum_condition(node, input_vals, scope)
        if isinstance(node, ProductNode):
            return prod_condition(node, input_vals, scope)
        if isinstance(node, LeafNode):
            return leaf_condition(node, input_vals, scope)


    scope = set([i for i in range(len(node.scope)) if not torch.isnan(evidence)[0][i]])
    new_root = recurse_pc(node, evidence, scope)[0]
    new_root = prune_pc(new_root)
    return new_root
