# graph_evaluator.py

from collections import deque
from typing import Any, Dict, List, Tuple

import networkx as nx

from .environment import Environment
from .models import Evaluator
from  .worfBench import t_eval_graph,match_node,SentenceTransformer

class GraphEvaluator:
    def __init__(
        self,
        incoming_graph_data,
        enable_shortcut: bool = False,
    ) -> None:
        self.G = nx.DiGraph(incoming_graph_data)
        assert nx.is_directed_acyclic_graph(self.G)
        self.total_nodes = self.G.number_of_nodes()
        self.complete_nodes = 0
        self.completeness = 0.0
        self.completeness_per_action = 0.0
        self.step_to_complete = self.G.number_of_edges()
        self.longest_unfinished_path_length = nx.dag_longest_path_length(self.G)
        self.enable_shortcut = enable_shortcut
        self.pred_graph = {
            "nodes": ["START"],
            "edges": [(0,1)]
        }
        self.gt_graph={
            "nodes":["START"],
            "edges":[]
        }
        i = 0
        for node in self.G.nodes():
            self.gt_graph["nodes"].append(node.name)
            self.gt_graph["edges"].append((i, i + 1))
            i += 1
        self.gt_graph["nodes"].append("END")
        self.gt_graph["edges"].append((i, i + 1))

        # Set the sink node for the DAG:
        sink_nodes = [node for node, out_degree in self.G.out_degree() if out_degree == 0]
        if len(sink_nodes) != 1:
            raise ValueError("Graph should have exactly one sink node.")
        self.sink_node = sink_nodes[0]

        self.human_mode = False

        self.reset()

    def reset(self):
        self.count = 0
        self.pred_graph = {
            "nodes": ["START"],
            "edges": [(0,1)]
        }
        for node in self.G.nodes():
            self.G.nodes[node]["remaining_predecessors"] = self.G.in_degree(node)
            self.G.nodes[node]["passing_count"] = None

    def step(
        self,
        envs: dict[str, Environment],
        default_env: str = "root",
    ):
        if self.is_complete():
            raise ValueError(
                "GraphEvaluator has already completed and "
                "cannot perform another step."
            )
        run_evaluators = set()
        evaluators = self.get_next_source_nodes()
        while evaluators:
            for evaluator in evaluators:
                # print("evaluator:", evaluator)
                if evaluator.local and self.human_mode:
                    result = True
                else:
                    environment = envs[evaluator.env_name or default_env]
                    result = environment.take_action(evaluator)
                # print("result:", result)
                # 添加到预测图
                if len(self.pred_graph["nodes"]) > 1:
                    last_node_index = len(self.pred_graph["nodes"])-1
                    # 如果是重复检查同一个节点，添加自循环边
                    if evaluator.name == self.pred_graph["nodes"][-1]:
                        self.pred_graph["edges"].append((last_node_index, last_node_index))
                    else:
                        self.pred_graph["edges"].append((last_node_index, last_node_index + 1))
                if evaluator.name not in self.pred_graph["nodes"]:
                    self.pred_graph["nodes"].append(evaluator.name)

                if result:
                    self.G.nodes[evaluator]["passing_count"] = self.count
                    self.complete_nodes += 1
                    for _, out_node in self.G.out_edges(evaluator):
                        self.G.nodes[out_node]["remaining_predecessors"] -= 1
            if self.is_complete():
                self.complete_nodes = self.total_nodes
                break
            run_evaluators.update(evaluators)
            evaluators = self.get_next_source_nodes() - run_evaluators

        self.update()

    def get_next_source_nodes(self) -> set[Evaluator]:
        if not self.enable_shortcut:
            source_nodes = []
            for node in self.G.nodes(data=True):
                if node[1]["passing_count"] is None and node[1]["remaining_predecessors"] == 0:
                    source_nodes.append(node[0])
        else:
            source_nodes = list(self.G.nodes())

        return set(source_nodes)

    def entry(self) -> bool:
        return all(count is not None for _, count in self.G.nodes(data="passing_count"))

    def update(self):
        self.count += 1
        self.completeness = float(self.complete_nodes / self.total_nodes)
        self.completeness_per_action = self.completeness / self.count

        self.step_to_complete = self.calculate_step_to_complete()
        self.longest_unfinished_path_length = self.calculate_longest_unfinished_path_length()

    def calculate_longest_unfinished_path_length(self) -> int:
        longest_path_length = 0
        if self.G.nodes[self.sink_node]["passing_count"] is not None:
            return longest_path_length

        visited = set()
        queue = deque([[self.sink_node]])

        while queue:
            path = queue.popleft()
            node = path[0]
            visited.add(node)
            longest_path_length = max(len(path), longest_path_length) - 1

            for predecessor in self.G.predecessors(node):
                if self.G.nodes[predecessor]["passing_count"] is not None:
                    continue
                elif predecessor not in visited:
                    queue.append([predecessor] + path)

        return longest_path_length

    def calculate_step_to_complete(self) -> int:
        incomplete_edges = 0
        if self.G.nodes[self.sink_node]["passing_count"] is not None:
            return incomplete_edges

        visited = set()
        queue = deque([self.sink_node])

        while queue:
            node = queue.popleft()
            visited.add(node)

            incomplete_edges += len(list(self.G.predecessors(node)))

            for predecessor in self.G.predecessors(node):
                if self.G.nodes[predecessor]["passing_count"] is not None:
                    continue
                elif predecessor not in visited:
                    queue.append(predecessor)

        return incomplete_edges

    def is_complete(self) -> bool:
        return self.G.nodes[self.sink_node]["passing_count"] is not None

    def get_completeness(self) -> float:
        return self.completeness

    def get_completeness_per_action(self) -> float:
        return self.completeness_per_action

    def get_step_to_complete(self) -> int:
        return self.step_to_complete

    def get_longest_unfinished_path_length(self) -> int:
        return self.longest_unfinished_path_length

    def convert_nodes_to_indices(graph: Dict[str, List[str]]) -> Dict[str, List]:
        # 创建节点到索引的映射
        node_to_index = {node: idx for idx, node in enumerate(graph["nodes"])}

        # 转换节点为索引
        nodes = list(range(len(graph["nodes"])))

        # 转换边为索引
        edges = [(node_to_index[u], node_to_index[v]) for u, v in graph["edges"]]

        return {"nodes": nodes, "edges": edges}

    def stat(self) -> dict[str, Any]:
        # print(self.pred_graph["nodes"])
        metrics={}
        if len(self.pred_graph["nodes"])>0:
            # 计算新增的评估指标
            metrics = self.compute_custom_metrics()
        # print("pred graph:",self.pred_graph)
        # 整合原有指标和新增指标
        result = {
            "total_nodes": self.total_nodes,
            "complete_nodes": self.complete_nodes,
            "completeness": self.completeness,
            "completeness_per_action": self.completeness_per_action,
            "step_to_complete": self.step_to_complete,
            "longest_unfinished_path_length": self.longest_unfinished_path_length,
        }
        result.update(metrics)

        return result

    def _check_submit(self, environment: Environment) -> bool:
        if not environment.trajectory:
            return False
        last_action = environment.trajectory[-1]
        if last_action[0] != "_submit":
            return False

        return last_action[2]

    def compute_radar_stats(self) -> dict[str, float]:
        longest_path_length = nx.dag_longest_path_length(self.G)
        return {
            "Completeness": float(self.completeness),
            "Efficiency": float(self.completeness_per_action),
            "Path Completeness Ratio": (
                longest_path_length - self.longest_unfinished_path_length
            )
            / longest_path_length,
        }

    @staticmethod
    def visualize(evaluators: list["GraphEvaluator"], path: str):
        import plotly.graph_objects as go

        fig = go.Figure()
        for i, evaluator in enumerate(evaluators):
            radar_stats = evaluator.compute_radar_stats()
            fig.add_trace(
                go.Scatterpolar(
                    r=list(radar_stats.values()),
                    theta=list(radar_stats.keys()),
                    fill="toself",
                    name=f"Graph Evaluator {i}",
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
        )
        fig.update_layout(
            margin=dict(l=150, r=150, t=150, b=150),
        )
        fig.write_image(path, scale=12, width=600, height=600)

    def compute_custom_metrics(self):
        # print("gt_graph:",self.gt_graph)

        # 加载 SentenceTransformer 模型
        eval_model = "all-mpnet-base-v2"
        eval_model = SentenceTransformer(eval_model)

        # 计算节点匹配
        pred_to_gt_mapping = match_node(self.pred_graph["nodes"], self.gt_graph["nodes"], eval_model)

        # 计算图评估指标
        graph_metrics = t_eval_graph(self.pred_graph, self.gt_graph, eval_model)


        # 整合所有指标
        metrics = {
            "precision": graph_metrics["precision"],
            "recall": graph_metrics["recall"],
            "f1_score": graph_metrics["f1_score"],
            "backtracking_rate": graph_metrics["backtracking_rate"]
        }

        return metrics