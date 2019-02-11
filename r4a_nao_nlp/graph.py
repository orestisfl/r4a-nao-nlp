# TODO: docstrings
# vim:ts=4:sw=4:expandtab:fo-=t
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import networkx as nx

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import Token


class Graph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_edge(self, a, words, b=None, words_after=None, *args, **kwargs):
        super().add_edge(a, b, label=words, *args, **kwargs)
        if words_after:
            assert b

            super().add_edge(b, None, label=words_after, *args, **kwargs)

    # TODO: Nodes in same height + vertical height?
    def plot(self, filename: str = "out.pdf") -> None:
        # TODO: catch import exceptions
        import matplotlib.pyplot as plt
        from adjustText import adjust_text

        initial_pos = {}
        fixed = []
        if None in self:
            initial_pos[None] = (10, 10)  # TODO
        for node, idx in nx.get_node_attributes(self, "idx_main").items():
            if idx == 0:
                fixed.append(node)
            initial_pos[node] = (idx, 0)

        pos = nx.spring_layout(self, pos=initial_pos, fixed=fixed)
        node_collection = nx.draw_networkx_nodes(self, pos)
        edge_collection = nx.draw_networkx_edges(self, pos)
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
                labels={
                    n: str(n.parsed) if n is not None else "" for n in self.nodes()
                },
            ).values()
        )

        plt.axis("off")
        adjust_text(texts, objects=[node_collection, edge_collection])

        plt.savefig(filename)
        plt.close()
        return node_collection, edge_collection


def _data_label(data: Dict[str, List[Token]]) -> str:
    return (
        "".join(t.text_with_ws for t in data["label"]).strip()
        if "label" in data
        else ""
    )
