import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import asyncio

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.gnn.gcn import GCN, MLP
from torch_geometric.utils import dense_to_sparse

class RefineModule(nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        self.layer1 = nn.Linear(num_nodes, num_nodes)
        self.layer2 = nn.Linear(num_nodes, num_nodes)
        self.activation = nn.ReLU()
        self.output = nn.Sigmoid()

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.layer1(adjacency)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.output(x)
        return 0.5 * (x + x.transpose(0, 1))

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        self.hidden_dim = 32
        self.latent_dim = 16
        self.gumbel_tau = 0.5

        input_dim = self.features.size(1) * 2
        self.gcn = GCN(input_dim, self.hidden_dim, self.hidden_dim)
        self.mlp = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.qs_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.refine = RefineModule(self.num_nodes)

        self.mu: Optional[torch.Tensor] = None
        self.logvar: Optional[torch.Tensor] = None
        self.latent_z: Optional[torch.Tensor] = None
        self.tilde_S: Optional[torch.Tensor] = None

        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks
        self.spatial_edge_probs = torch.full((len(self.potential_spatial_edges),),
                                             initial_spatial_probability,
                                             dtype=torch.float32)

        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
        self.temporal_edge_probs = torch.full((len(self.potential_temporal_edges),),
                                              initial_temporal_probability,
                                              dtype=torch.float32)
    
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
        new_features = torch.cat((self.features,query_embedding),dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        device = self.spatial_edge_probs.device
        log_probs = [torch.tensor(0.0, device=device)]

        for idx, (potential_connection, edge_mask) in enumerate(zip(self.potential_spatial_edges, self.spatial_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = self.spatial_edge_probs[idx]
                edge_prob = torch.sigmoid(torch.logit(edge_prob, eps=1e-6) / temperature)
                if threshold:
                    hard_value = 1.0 if edge_prob.item() > threshold else 0.0
                    edge_prob = torch.tensor(hard_value, dtype=edge_prob.dtype, device=device)
                edge_prob = torch.clamp(edge_prob, min=1e-6, max=1 - 1e-6)
                bernoulli = Bernoulli(probs=edge_prob)
                sample = bernoulli.sample()
                if sample.item() == 1.0:
                    out_node.add_successor(in_node,'spatial')
                log_probs.append(bernoulli.log_prob(sample))

        return torch.sum(torch.stack(log_probs))

    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        device = self.temporal_edge_probs.device
        log_probs = [torch.tensor(0.0, device=device)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))
        for idx, (potential_connection, edge_mask) in enumerate(zip(self.potential_temporal_edges, self.temporal_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue

            edge_prob = self.temporal_edge_probs[idx]
            edge_prob = torch.sigmoid(torch.logit(edge_prob, eps=1e-6) / temperature)
            if threshold:
                hard_value = 1.0 if edge_prob.item() > threshold else 0.0
                edge_prob = torch.tensor(hard_value, dtype=edge_prob.dtype, device=device)
            edge_prob = torch.clamp(edge_prob, min=1e-6, max=1 - 1e-6)
            bernoulli = Bernoulli(probs=edge_prob)
            sample = bernoulli.sample()
            if sample.item() == 1.0:
                out_node.add_successor(in_node,'temporal')
            log_probs.append(bernoulli.log_prob(sample))

        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        if isinstance(inputs, dict) and 'task' in inputs:
            self.prepare_probabilities(inputs['task'])

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.prepare_probabilities(input['task'])

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs

    def encode(self, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        new_features = self.construct_new_features(query)
        latent_features = self.gcn(new_features, self.role_adj_matrix)
        latent_features = self.mlp(latent_features)
        mu = self.encoder_mu(latent_features)
        logvar = self.encoder_logvar(latent_features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def qs(self, latent: torch.Tensor, tau: float) -> torch.Tensor:
        transformed = self.qs_linear(latent)
        similarity = torch.matmul(transformed, transformed.transpose(0, 1))
        if self.qs_linear.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity) + 1e-9) + 1e-9)
            logits = (similarity + gumbel_noise) / tau
            return torch.sigmoid(logits)
        return torch.sigmoid(similarity)

    def qc(self, sketch_adj: torch.Tensor) -> torch.Tensor:
        refined = self.refine(sketch_adj)
        return refined

    def prepare_probabilities(self, query: str) -> None:
        self.mu, self.logvar = self.encode(query)
        self.latent_z = self.reparameterize(self.mu, self.logvar)
        sketch_adj = self.qs(self.latent_z, self.gumbel_tau)
        self.tilde_S = self.qc(sketch_adj)
        flat_probs = torch.clamp(self.tilde_S, min=1e-6, max=1 - 1e-6).reshape(-1)
        self.spatial_edge_probs = flat_probs
        self.temporal_edge_probs = flat_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            if num_edges.item() > 0:
                prune_target = torch.round(num_edges * pruning_rate)
                prune_num_edges = int(prune_target.item()) if prune_target.item() > 0 else 1
                edge_scores = self.spatial_edge_probs.clone().detach()
                edge_scores = torch.where(self.spatial_masks > 0, edge_scores, torch.tensor(1.0, dtype=edge_scores.dtype, device=edge_scores.device))
                sorted_edges_idx = torch.argsort(edge_scores)
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks.item())]
                self.spatial_masks[prune_idx] = 0

        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            if num_edges.item() > 0:
                prune_target = torch.round(num_edges * pruning_rate)
                prune_num_edges = int(prune_target.item()) if prune_target.item() > 0 else 1
                edge_scores = self.temporal_edge_probs.clone().detach()
                edge_scores = torch.where(self.temporal_masks > 0, edge_scores, torch.tensor(1.0, dtype=edge_scores.dtype, device=edge_scores.device))
                sorted_edges_idx = torch.argsort(edge_scores)
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks.item())]
                self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
