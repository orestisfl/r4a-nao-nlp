# TODO: docstrings
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import networkx as nx

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import Token, Node, EObject
    from numpy import ndarray


class Graph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        self.sent_idx = 0
        super().__init__(*args, **kwargs)
        self.add_node("Start")

    def add_edge(
        self, a: Node, words: List[Token], b: Optional[Node] = None, **kwargs
    ) -> None:
        if b is None:
            # Make sure that sent_end is in the graph with the correct sent_idx
            b = self.sent_end
            self.add_node(b)

        super().add_edge(a, b, label=words, **kwargs)

    def add_node(self, node: Node, **kwargs) -> None:
        kwargs.setdefault("sent_idx", self.sent_idx)
        super().add_node(node, **kwargs)

    @property
    def prev_end(self) -> str:
        return self._sent_end(self.sent_idx - 1) if self.sent_idx > 0 else "Start"

    @property
    def sent_end(self) -> str:
        return self._sent_end(self.sent_idx)

    @staticmethod
    def _sent_end(idx) -> str:
        return f"End-{idx}"

    def connect_prev(self, node: Node, words: List[Token]) -> None:
        self.add_edge(self.prev_end, words, node)

    def to_pydot(self, filename: str = "out.gv") -> None:
        """Convert graph to DOT language and save"""
        # TODO: escape underscores
        g = nx.relabel_nodes(self, self.node_label_mapping)
        # Use boxes for subsentences
        for _, d in g.nodes(data=True):
            if "idx" not in d:
                continue
            d["shape"] = "box"
        # Always quote labels to avoid issues
        for u, v, d in g.edges(data=True):
            g[u][v]["label"] = '"' + _data_label(d) + '"'
        with open(filename, "w") as f:
            nx.drawing.nx_pydot.write_dot(g, f)

    # TODO: Nodes in same height + vertical height?
    # TODO: Return
    def plot(self, filename: str = "out.pdf"):
        # TODO: catch import exceptions
        import matplotlib.pyplot as plt
        from adjustText import adjust_text

        pos = self._create_pos()
        node_collection = nx.draw_networkx_nodes(self, pos, node_size=50)
        edge_collection = nx.draw_networkx_edges(self, pos, arrowstyle="->")
        # TODO: also include information about ARGMs types
        nx.draw_networkx_edge_labels(
            self,
            pos,
            font_size=8,
            edge_labels={(u, v): _data_label(d) for u, v, d in self.edges(data=True)},
        )
        texts = list(
            nx.draw_networkx_labels(
                self,
                # TODO: better text positioning
                {key: value + (0.0, 0.08) for key, value in pos.items()},
                font_size=6,
                labels=self.node_label_mapping,
            ).values()
        )

        plt.axis("off")
        adjust_text(texts, objects=[node_collection, edge_collection])

        plt.savefig(filename)
        plt.close()
        return node_collection, edge_collection

    @property
    def node_label_mapping(self):
        return {
            n: n
            if isinstance(n, str)
            else (
                f"Transition({n.parsed.input})"
                if n.parsed.name == "Transition"
                else str(n.parsed)
            )
            for n in self.nodes()
        }

    def _create_pos(self) -> Dict[Node, ndarray]:
        import numpy

        result = {}
        max_x = 1 + max(nx.get_node_attributes(self, "idx").values())
        for node, data in self.nodes(data=True):
            if isinstance(node, str):
                x = max_x
                is_mod = False
            else:
                x = data["idx"]
                is_mod = "idx_main" not in data
            y = -(2 * data["sent_idx"] + is_mod)
            result[node] = numpy.array((x, y))
        result["Start"] = numpy.array((0, 1))
        return result

    def to_eobject(self, name: Optional[str] = None) -> EObject:
        from r4a_nao_nlp import ecore

        return ecore.naoapp_from_nodes(self, name)


def _data_label(data: Dict[str, List[Token]]) -> str:
    tokens = data.get("label") or []
    return "".join(t.text_with_ws for t in tokens).strip()


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
