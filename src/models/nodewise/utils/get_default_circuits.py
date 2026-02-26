import torch
import random

from ..nodes import SumNode, ProductNode, LeafNode, Node
from ..distributions import Gaussian



def get_default_circuit_sum_root(loc=1.0, offset=0.0):
    x1 = Gaussian(scope=[0], mean=torch.tensor(-loc+offset), log_stdev=torch.tensor(0.0))
    x2 = Gaussian(scope=[0], mean=torch.tensor( loc+offset), log_stdev=torch.tensor(0.0))
    y1 = Gaussian(scope=[1], mean=torch.tensor(-loc+offset), log_stdev=torch.tensor(0.0))
    y2 = Gaussian(scope=[1], mean=torch.tensor( loc+offset), log_stdev=torch.tensor(0.0))
    prod1 = ProductNode(scope=[0, 1], children_circuit=[x1, y1])
    prod2 = ProductNode(scope=[0, 1], children_circuit=[x2, y2])
    sum = SumNode(scope=[0, 1], children_circuit=[prod1, prod2], log_weights=torch.log(torch.tensor([0.5, 0.5])))
    sum.is_valid()
    return sum


def get_default_circuit_prod_root():
    x1 = Gaussian(scope=[0], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
    x2 = Gaussian(scope=[0], mean=torch.tensor( 1.0), log_stdev=torch.tensor(0.0))
    y1 = Gaussian(scope=[1], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
    y2 = Gaussian(scope=[1], mean=torch.tensor( 1.0), log_stdev=torch.tensor(0.0))
    sum1 = SumNode(scope=[0], children_circuit=[x1, x2], log_weights=torch.log(torch.tensor([0.5, 0.5])))
    sum2 = SumNode(scope=[1], children_circuit=[y1, y2], log_weights=torch.log(torch.tensor([0.5, 0.5])))
    prod = ProductNode(scope=[0, 1], children_circuit=[sum1, sum2])
    prod.is_valid()
    return prod


def get_3d_circuit():
    x1 = Gaussian(scope=[0], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
    x2 = Gaussian(scope=[0], mean=torch.tensor( 1.0), log_stdev=torch.tensor(0.0))
    y1 = Gaussian(scope=[1], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
    y2 = Gaussian(scope=[1], mean=torch.tensor( 1.0), log_stdev=torch.tensor(0.0))
    z1 = Gaussian(scope=[2], mean=torch.tensor(-1.0), log_stdev=torch.tensor(0.0))
    z2 = Gaussian(scope=[2], mean=torch.tensor( 1.0), log_stdev=torch.tensor(0.0))
    prod1 = ProductNode(scope=[0, 1, 2], children_circuit=[x1, y1, z1])
    prod2 = ProductNode(scope=[0, 1, 2], children_circuit=[x2, y2, z2])
    sum = SumNode(scope=[0, 1, 2], children_circuit=[prod1, prod2], log_weights=torch.log(torch.tensor([0.5, 0.5])))
    sum.is_valid()
    return sum



def generate_random_pc(num_variables: int, max_depth: int, leafnode: type) -> Node:
    """
    Generate a random tree-structured Probabilistic Circuit.
    
    Args:
        num_variables: Number of random variables (scopes)
        max_depth: Maximum depth of the tree
        
    Returns:
        Root node of the generated circuit (always a SumNode)
    """
    all_scope = list(range(num_variables))
    
    def split_scope(scope: list[int]) -> list[list[int]]:
        """Split scope into disjoint subsets for ProductNode children."""
        if len(scope) <= 1:
            return [scope]
        
        # Randomly split scope into 2-4 parts
        num_splits = min(len(scope), random.randint(2, 4))
        shuffled = scope.copy()
        random.shuffle(shuffled)
        
        splits = []
        split_size = len(scope) // num_splits
        remainder = len(scope) % num_splits
        
        start = 0
        for i in range(num_splits):
            # Distribute remainder across first splits
            current_size = split_size + (1 if i < remainder else 0)
            if current_size > 0:
                splits.append(shuffled[start:start + current_size])
                start += current_size
        
        return [s for s in splits if len(s) > 0]
    
    def build_node(scope: list[int], current_depth: int, parent_type: str) -> Node:
        """
        Recursively build the circuit tree.
        
        Args:
            scope: list of variable indices for this node
            current_depth: Current depth in the tree
            parent_type: Type of parent node ('sum' or 'product')
            
        Returns:
            A Node (SumNode, ProductNode, or LeafNode)
        """
        # Base case: create leaf nodes at max depth
        # Leaf nodes must have scope length of 1
        if current_depth >= max_depth or len(scope) == 0:
            if len(scope) == 1:
                # Single variable: create 1-3 leaf nodes
                num_leaves = random.randint(1, 3)
                leaves = [leafnode(scope=scope.copy()) for _ in range(num_leaves)]
                
                # If multiple leaves, wrap them in a SumNode
                if len(leaves) > 1:
                    return SumNode(scope=scope.copy(), children_circuit=leaves)
                else:
                    return leaves[0]
            else:
                # Multiple variables: must create ProductNode to split them
                # This ensures each leaf gets only one variable
                children = []
                for s in scope:
                    num_leaves = random.randint(1, 3)
                    leaves = [leafnode(scope=[s]) for _ in range(num_leaves)]
                    
                    # Wrap multiple leaves in SumNode
                    if len(leaves) > 1:
                        children.append(SumNode(scope=[s], children_circuit=leaves))
                    else:
                        children.append(leaves[0])
                
                return ProductNode(scope=scope.copy(), children_circuit=children)
        
        # Alternate between Sum and Product nodes
        node_type = 'product' if parent_type == 'sum' else 'sum'
        
        if node_type == 'sum':
            # SumNode: create 2-4 children with SAME scope (smoothness)
            num_children = random.randint(2, 4)
            children = []
            
            for _ in range(num_children):
                child = build_node(scope.copy(), current_depth + 1, 'sum')
                children.append(child)
            
            return SumNode(scope=scope.copy(), children_circuit=children)
        
        else:  # node_type == 'product'
            # ProductNode: split scope into DISJOINT subsets (decomposability)
            scope_splits = split_scope(scope)
            children = []
            
            for sub_scope in scope_splits:
                child = build_node(sub_scope, current_depth + 1, 'product')
                children.append(child)
            
            return ProductNode(scope=scope.copy(), children_circuit=children)
    
    # Start with root SumNode (parent_type='product' ensures we get 'sum')
    root = build_node(all_scope, 0, 'product')
    return root



def generate_random_pc_binary(num_variables: int, max_depth: int, leafnode: type, leaves_parameters: dict[str, float] = {}) -> Node:
    """
    Generate a random tree-structured Probabilistic Circuit with binary splits.
    
    Args:
        num_variables: Number of random variables (scopes)
        max_depth: Maximum depth of the tree
        
    Returns:
        Root node of the generated circuit (always a SumNode)
        
    Properties:
        - Binary tree: all nodes have exactly 2 children
        - All leaves have scope length of 1
        - Smooth and decomposable
    """
    all_scope = list(range(num_variables))
    
    def split_scope_binary(scope: list[int]) -> tuple[list[int], list[int]]:
        """Split scope into two disjoint subsets."""
        if len(scope) == 1:
            return [scope[0]], []
        
        # Randomly shuffle and split in half
        shuffled = scope.copy()
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        
        # Ensure at least one variable in each split if possible
        if mid == 0:
            mid = 1
        
        left = shuffled[:mid]
        right = shuffled[mid:]
        
        return left, right
    
    def build_leaf_product(scope: list[int]) -> Node:
        """Helper to build ProductNode of leaves for scopes > 1."""
        if len(scope) == 1:
            return leafnode(scope=scope.copy())
        
        left, right = split_scope_binary(scope)
        if len(right) == 0:
            return leafnode(scope=left)
        
        left_child = leafnode(scope=left) if len(left) == 1 else build_leaf_product(left)
        right_child = leafnode(scope=right) if len(right) == 1 else build_leaf_product(right)
        
        return ProductNode(scope=scope.copy(), children_circuit=[left_child, right_child])
    
    def build_node(scope: list[int], current_depth: int, parent_type: str) -> Node:
        """
        Recursively build the circuit tree with binary splits.
        
        Args:
            scope: List of variable indices for this node
            current_depth: Current depth in the tree
            parent_type: Type of parent node ('sum' or 'product')
            
        Returns:
            A Node (SumNode, ProductNode, or LeafNode)
        """
        # Base case: at max depth, create leaves
        if current_depth >= max_depth:
            if len(scope) == 1:
                # Single variable: create a leaf node
                return leafnode(scope=scope.copy(), **leaves_parameters)
            else:
                # Multiple variables at max depth: must split into leaves
                # Create a ProductNode to maintain decomposability
                left, right = split_scope_binary(scope)
                
                if len(right) == 0:
                    return leafnode(scope=left, **leaves_parameters)
                
                left_child = leafnode(scope=left, **leaves_parameters) if len(left) == 1 else build_leaf_product(left)
                right_child = leafnode(scope=right, **leaves_parameters) if len(right) == 1 else build_leaf_product(right)
                
                return ProductNode(scope=scope.copy(), children_circuit=[left_child, right_child])
        
        # If single variable before max depth, create leaf early
        if len(scope) == 1:
            return leafnode(scope=scope.copy(), **leaves_parameters)
        
        # Alternate between Sum and Product nodes
        node_type = 'product' if parent_type == 'sum' else 'sum'
        
        if node_type == 'sum':
            # SumNode: create exactly 2 children with SAME scope (smoothness)
            child1 = build_node(scope.copy(), current_depth + 1, 'sum')
            child2 = build_node(scope.copy(), current_depth + 1, 'sum')
            
            return SumNode(scope=scope.copy(), children_circuit=[child1, child2])
        
        else:  # node_type == 'product'
            # ProductNode: split scope into 2 DISJOINT subsets (decomposability)
            left_scope, right_scope = split_scope_binary(scope)
            
            if len(right_scope) == 0:
                # Only one variable, can't split further
                return build_node(left_scope, current_depth + 1, 'product')
            
            left_child = build_node(left_scope, current_depth + 1, 'product')
            right_child = build_node(right_scope, current_depth + 1, 'product')
            
            return ProductNode(scope=scope.copy(), children_circuit=[left_child, right_child])
    
    # Start with root SumNode (parent_type='product' ensures we get 'sum')
    if num_variables > 1:
        root = build_node(all_scope, 0, 'product')
    else:
        root = leafnode(scope=[0], **leaves_parameters)
    return root



# Example usage:
if __name__ == "__main__":
    # Generate a random PC with 4 variables and depth 3
    circuit = generate_random_pc(num_variables=10, max_depth=4, leafnode=Gaussian)
    print(circuit)
    
    # Print circuit statistics
    def count_nodes(node):
        if isinstance(node, LeafNode):
            return {'sum': 0, 'product': 0, 'leaf': 1}
        
        counts = {'sum': 0, 'product': 0, 'leaf': 0}
        if isinstance(node, SumNode):
            counts['sum'] = 1
        elif isinstance(node, ProductNode):
            counts['product'] = 1
        
        for child in node.children_circuit:
            child_counts = count_nodes(child)
            for key in counts:
                counts[key] += child_counts[key]
        
        return counts
    
    stats = count_nodes(circuit)
    print(f"Generated Circuit Statistics:")
    print(f"  Sum Nodes: {stats['sum']}")
    print(f"  Product Nodes: {stats['product']}")
    print(f"  Leaf Nodes: {stats['leaf']}")
    print(f"  Total Nodes: {sum(stats.values())}")
    print(f"  Root scope: {circuit.scope}")


    with torch.no_grad():
        print(f"Mean: {circuit.mean()}")
        print(f"Var : {circuit.var()}")
        print(f"Std : {circuit.std()}")



