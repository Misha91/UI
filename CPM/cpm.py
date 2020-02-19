# -*- coding: utf-8 -*-
"""
    CPM.py
    ~~~~~~
    Critical path method

    :author: Jiri Spilka, 2019
"""

import networkx as nx
import matplotlib.pyplot as plt


class CPM(nx.DiGraph):
    """
    Graph representation subclassing networkx.DiGraph
    """

    def __init__(self):
        super().__init__()
        self._makespan = -1
        self._critical_path = None
        self.Cj = {}
        self.Cjp = {}
        self.Sj = {}
        self.Sjp = {}

    def _forward(self):
        """
        Step 1: For each job j that has no predecessors Sj = 0 and Cj = pj
        Step 2: Compute inductively for each remaining job j
            Sj = max Ck_{all k - j}
            Cj = Sj + pj
        Step 3: Cmax = max(C1,..., Cn)

        :return:
        """


        Sj = {}
        Cj = {}

        for c in nx.topological_sort(self):

            if len(list(self.predecessors(c))) == 0:
                Sj[c] = 0
                Cj[c] = nx.get_node_attributes(self, 'p')[c]

        for c in nx.topological_sort(self):
            try:
                if (Sj[c] == 0):
                   pass
            except KeyError:
                Sj[c] = max([Cj[j] for j in self.predecessors(c)])
                Cj[c] = Sj[c] + nx.get_node_attributes(self, 'p')[c]

        self.Cj = Cj
        self.Sj = Sj

        self._makespan = max(Cj.values())

    def _backward(self):
        """
        Step 1: For each job j that has no successors Cj = Cmax and Sj = Cmax − pj
        Step 2: Compute inductively for each remaining job j
                Cj = min Sk{ j - all k }
                Sj = Cj − pj
        Step 3: Verify that 0 = min(S1, ... ,Sn)

        :return:
        """
        #raise NotImplemented
        Sjp = {}
        Cjp = {}
        # print("no pred")
        for c in list(reversed(list(nx.topological_sort(self)))):

            if len(list(self.successors(c))) == 0:
                # print(c, (list(self.predecessors(c))), len(list(self.predecessors(c))))
                Cjp[c] = self._makespan
                Sjp[c] = Cjp[c] - nx.get_node_attributes(self, 'p')[c]

            else:
                Cjp[c] = min([Sjp[j] for j in self.successors(c)])
                Sjp[c] = Cjp[c] - nx.get_node_attributes(self, 'p')[c]

        if min(Sjp.values()) != 0:
            raise Exception("!!! min(Sj'') is not equal to 0 !!!")
        self.Sjp = Sjp
        self.Cjp = Cjp
        #print(Sjp, Cjp)

    def _compute_critical_path(self):
        """
        The jobs whose earliest possible starting times are equal to their latest possible starting times are
        critical jobs.
        A critical path is a chain of jobs which begin at time 0 and ends at Cmax.

        :return:
        """
        tmp_path = []
        for s in self.Sj:
            if self.Sj[s] == self.Sjp[s]:
                tmp_path.append(s)

        for n in tmp_path:
            if (self.Sj[n] + nx.get_node_attributes(self, 'p')[n] != self.Cj[n]):
                raise Exception("Incorrect path!")

        if (self.Sj[tmp_path[0]] != 0 or self.Cjp[tmp_path[-1]] != self._makespan):
            raise Exception("Incorrect path!")


        self._critical_path = self.subgraph(tmp_path)


    @property
    def makespan(self):
        self.compute()
        return self._makespan

    @property
    def critical_path(self):
        self.compute()
        return self._critical_path

    def compute(self):
        self._forward()
        self._backward()
        self._compute_critical_path()

    def draw_graph(self):

        pos = nx.kamada_kawai_layout(self)  # positions for all nodes

        labels = {v: str(v) for v in self.nodes}
        nx.draw_networkx_nodes(self, pos, alpha=0.2)
        nx.draw_networkx_edges(self, pos)
        nx.draw_networkx_labels(self, pos, labels, font_size=16)

        plt.draw()
        plt.show()


if __name__ == "__main__":

    cpm = CPM()

    cpm.add_node(1, p=4)
    cpm.add_node(2, p=9)
    cpm.add_node(3, p=3)
    cpm.add_node(4, p=3)
    cpm.add_node(5, p=6)
    cpm.add_node(6, p=8)
    cpm.add_node(7, p=8)
    cpm.add_node(8, p=12)
    cpm.add_node(9, p=6)

    cpm.add_edges_from([(1, 2), (2, 6), (6, 7)])
    cpm.add_edges_from([(3, 4), (4, 5), (5, 6), (5, 8)])
    cpm.add_edges_from([(8, 7), (8, 9)])

    print('nodes in the graph')
    print(cpm.nodes)

    print('Cmax (makespan)')
    print(cpm.makespan)

    print('nodes on the critical path')
    print(cpm.critical_path.nodes())

    cpm.draw_graph()