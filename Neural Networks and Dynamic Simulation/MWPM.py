#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Generates circuits for quantum error correction with surface code patches.
"""
import copy
import warnings

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

try:
    from qiskit import Aer

    HAS_AER = True
except ImportError:
    from qiskit import BasicAer

    HAS_AER = False


class SurfaceCode:
    """
    Implementation of a distance d surface code, implemented over
    T syndrome measurement rounds.
    """

    def __init__(self, d, T):
        """
        Creates the circuits corresponding to a logical 0 encoded
        using a surface code with X and Z stabilizers.
        Args:
            d (int): Number of physical "data" qubits. Only odd d's allowed
            T (int): Number of rounds of ancilla-assisted syndrome measurement. Normally T=d
        Additional information:
            No measurements are added to the circuit if `T=0`. Otherwise
            `T` rounds are added, followed by measurement of the code
            qubits (corresponding to a logical measurement and final
            syndrome measurement round)
            This circuit is for "rotated lattices" i.e. it requires
            d**2 data qubits and d**2-1 syndrome qubits only. Hence,
            d=odd allows equal number of Z and X stabilizer mesaurments.
        """
        self.d = d
        self.T = 0

        self.data = QuantumRegister(d ** 2, "data")
        self.ancilla = QuantumRegister((d ** 2 - 1), "ancilla")
        self.c_output = ClassicalRegister(d ** 2, "c_output")

        self.output = []
        self.circuit = {}

        for logical in ["0", "1"]:
            self.circuit[logical] = QuantumCircuit(
                self.ancilla, self.data, name="logical-{}".format(logical)
            )

        #        self._preparation() to be included to create log='1'

        for _ in range(T - 1):
            self.syndrome_measurement()

        if T != 0:
            self.syndrome_measurement(reset=False)
            self.readout()

    """It assigns vertices to qubits on a 2D graph, where,
    data qubits are on the x lines and, syndrome qubits are
    on the 0.5+x lines, in the cartesian coordinate system, where x is an integer."""

    def lattice(self):
        d = self.d
        data_string = nx.Graph()
        syndrome_string = nx.Graph()
        for i in range(0, d):
            for j in range(0, d):
                data_string.add_node((i, j))
        for k in range(0, d, 1):
            for i in range(0, d + 1, 1):
                for j in range(0, d + 1, 1):
                    if (i + j) % 2 != 0:
                        if ((i % 2 == 0) and j != d) or ((i % 2 == 1) and (j != 0)):
                            syndrome_string.add_node(((2 * i - 1) / 2, (2 * j - 1) / 2))
                    else:
                        if ((j % 2 == 0) and i != 0) or ((j % 2 == 1) and (i != d)):
                            syndrome_string.add_node(((2 * i - 1) / 2, (2 * j - 1) / 2))

        syn_ind = list(syndrome_string.nodes)
        data_ind = list(data_string.nodes)
        return (syn_ind, data_ind)

    def connection(self):
        """
        Determines the order of syndrome measurements between data qubits and syndrome qubits.
        We follow the ZN rule here to avoid hook error as described by [https://doi.org/10.1063/1.1499754]
        where Z stabilisers are arranged in 'Z' pattern and X stabilizers in 'N' pattern.
        Refer to the diagram in readme to get the refrence.
        """
        syn_index, data_index = self.lattice()

        order = []
        for i in range(self.d ** 2 - 1):
            d = data_index
            r = syn_index[i][0]
            c = syn_index[i][1]

            def get_index(j):
                for i in range(len(data_index)):
                    if data_index[i] == j:
                        return i

            new = []
            new.append((r, c))
            if r == -0.5:  # top semicircile
                new.append(-1)
                new.append(get_index((r + 0.5, c - 0.5)))
                new.append(-1)
                new.append(get_index((r + 0.5, c + 0.5)))
            elif c == -0.5:  # left semicircle
                new.append(-1)
                new.append(get_index((r - 0.5, c + 0.5)))
                new.append(-1)
                new.append(get_index((r + 0.5, c + 0.5)))

            elif r == self.d - 0.5:  # bottom semicircle

                new.append(get_index((r - 0.5, c - 0.5)))
                new.append(-1)
                new.append(get_index((r - 0.5, c + 0.5)))
                new.append(-1)

            elif c == self.d - 0.5:  # right semicircle
                new.append(get_index((r - 0.5, c - 0.5)))
                new.append(-1)
                new.append(get_index((r + 0.5, c - 0.5)))
                new.append(-1)
            else:
                if (r + c) % 2 == 0:  # square patches
                    new.append(get_index((r - 0.5, c - 0.5)))
                    new.append(get_index((r + 0.5, c - 0.5)))
                    new.append(get_index((r - 0.5, c + 0.5)))
                    new.append(get_index((r + 0.5, c + 0.5)))
                else:
                    new.append(get_index((r - 0.5, c - 0.5)))
                    new.append(get_index((r - 0.5, c + 0.5)))
                    new.append(get_index((r + 0.5, c - 0.5)))
                    new.append(get_index((r + 0.5, c + 0.5)))
            order.append(new)
        return order

    def syndrome_measurement(self, reset=True, barrier=True):
        """
            Application of a syndrome measurement round.
            Args:
                reset (bool): If set to true add a boolean at the end of each round
                barrier (bool): Boolean denoting whether to include a barrier at the end.
                A barrier is included after every round of 'j' which passes through layers of
                cx to be done, because the order should not be disturbed else the stabilizers
                will not be executed since Z and X on the same qubit do not commute. Thus,
                we end up flipping the sign of some stabilizers.
            """
        self.output.append(
            ClassicalRegister((self.d ** 2 - 1), "round_" + str(self.T) + "ancilla")
        )

        for log in ["0", "1"]:
            self.circuit[log].add_register(self.output[-1])
            order = self.connection()
            for j in range(1, 5):
                for i in range(len(order)):
                    k = self.data[order[i][j]]
                    l = self.ancilla[i]
                    if (order[i][0][0] + order[i][0][1]) % 2 == 0:  # Xstabilizer
                        if j == 1:
                            self.circuit[log].h(l)
                        if order[i][j] != -1:
                            self.circuit[log].cx(l, k)
                        if j == 4:
                            self.circuit[log].h(l)
                    else:  # Xstabilizer
                        if order[i][j] != -1:
                            self.circuit[log].cx(k, l)
                if barrier:
                    self.circuit[log].barrier()

            for j in range(self.d ** 2 - 1):
                if (order[j][0][0] + order[j][0][1]) % 2 == 1:  # Z
                    self.circuit[log].measure(self.ancilla[j], self.output[self.T][j])
                    if reset:
                        self.circuit[log].reset(self.ancilla[j])

            self.circuit[log].barrier()

            for j in range(self.d ** 2 - 1):
                if (order[j][0][0] + order[j][0][1]) % 2 == 0:  # X
                    self.circuit[log].measure(self.ancilla[j], self.output[self.T][j])
                    if reset:
                        self.circuit[log].reset(self.ancilla[j])

            self.circuit[log].barrier()

        self.T += 1

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        for log in ["0", "1"]:
            self.circuit[log].add_register(self.c_output)
            for i in range(self.d ** 2):
                self.circuit[log].measure(self.data[i], self.c_output[i])

    def process_results(self, raw_results):
        """
        Args:
            raw_results (dict): A dictionary whose keys are logical values,
                and whose values are standard counts dictionaries, (as
                obtained from the `get_counts` method of a ``qiskit.Result``
                object).
        Returns:
            syn: d+1 dimensional array where 0th array stores qubit readouts
            while the subsequesnt rows store the results from measurement rounds
            as required for extraction of nodes with errors to be sent to the decoder
        Additional information:
            The circuits must be executed outside of this class, so that
            their is full freedom to compile, choose a backend, use a
            noise model, etc. The results from these executions should then
            be used to create the input for this method.
        """
        results = []
        results = list(max(raw_results, key=raw_results.get))

        syn = []
        new = []
        for i in results:
            for j in range(len(i)):
                if i[j] != " ":
                    new.append(int(i[j]))
                else:
                    syn.append(new)
                    new = []
        syn.append(new)

        return syn

    def extract_nodes(self, syn_meas_results):
        """Extracts node locations of qubits which flipped in
        consecutive rounds (stored as (k,i,j)) and the data qubits which were flipped
        during readout (stored as (-2,i,j)). Here k spans range(0,d-1,1)
        Z syndrome nodes and Z logical data qubit nodes (see figure) in error_nodesZ
        and we do the same for X stabilizers and X logical qubits in error_nodesX.
        Note that arrays are reversed in terms of syndrome rounds, when compared to
        syn_meas_results
        """
        processed_results = []
        new = []
        for j in syn_meas_results[0]:
            new.append(j)
        processed_results.append(new)
        new = []
        for j in syn_meas_results[len(syn_meas_results) - 1]:
            new.append(j)
        processed_results.append(new)

        for i in range(len(syn_meas_results) - 2, 0, -1):
            new = []
            for j in range(0, len(syn_meas_results[i])):
                new.append((syn_meas_results[i][j] + syn_meas_results[i + 1][j]) % 2)
            processed_results.append(new)

        syn, dat = self.lattice()
        error_nodesX = []
        error_nodesZ = []

        # first_row = processed_result[0][:self.d]
        # last_row = processed_result[0][-self.d - 1:-1]

        # left_col = processed_result[0][::self.d]
        # right_col = processed_result[0][self.d-1:-1:self.d]

        # if sum(first_row) % 2 == 1 or sum(last_row) % 2 == 1:
        #     for node in dat[:self.d]:
        #         # Append virtual node
        #         if node[1] == 0:
        #             error_nodesZ.append((-1, node[0] - 0.5, node[1] - 0.5))
        #         else:
        #             error_nodesZ.append((-1, node[0] - 0.5, node[1] + 0.5))

        #     for node in dat[-self.d - 1:-1]:
        #         if node[1] == self.d - 1:
        #             error_nodesZ.append((-1, node[0] + 0.5, node[1] + 0.5))
        #         else:
        #             error_nodesZ.append((-1, node[0] + 0.5, node[1] - 0.5))

        # if sum(left_col) % 2 == 1 or sum(right_col) % 2 == 1:
        #     for node in dat[::self.d]:
        #         error_nodesX.append((-2, node[0], node[1]))
        #     for node in dat[self.d-1:-1:self.d]:
        #         error_nodesX.append((-2, node[0], node[1]))

        for i in range(2, len(processed_results)):
            for j in range(len(processed_results[i])):

                if processed_results[i][j] == 1:

                    if (syn[j][0] + syn[j][1]) % 2 == 0:
                        error_nodesX.append((i - 1, syn[j][0], syn[j][1]))
                    else:
                        error_nodesZ.append((i - 1, syn[j][0], syn[j][1]))
        return error_nodesX, error_nodesZ


# In[4]:


from qiskit import QuantumCircuit, execute


try:
    from qiskit import Aer

    HAS_AER = True
except ImportError:
    from qiskit import BasicAer

    HAS_AER = False


class GraphDecoder:
    """
    Class to construct the graph corresponding to the possible syndromes
    of a quantum error correction code, and then run suitable decoders.
    """

    def __init__(self, d, T, simulation=False):

        self.d = d
        self.T = T
        self.virtual = self._specify_virtual()
        self.S = {"X": nx.Graph(), "Z": nx.Graph()}
        self.simulation = simulation
        if simulation:
            self.code = SurfaceCode(d, T)
            self._make_syndrome_graph_simulate()
        else:
            self._make_syndrome_graph()

    def _specify_virtual(self):
        """Define coordinates of Z and X virtual nodes. Our convention is that Z
        virtual nodes are top/bottom and X virtual nodes are left/right.
        """
        virtual = {}
        virtual["X"] = []
        virtual["Z"] = []
        for j in range(0, self.d, 2):
            # Z virtual nodes
            virtual["Z"].append((-1, -0.5, j - 0.5))  # top
            virtual["Z"].append((-1, self.d - 0.5, j + 0.5))  # bottom

            # X virtual nodes
            virtual["X"].append((-1, j + 0.5, -0.5))  # left
            virtual["X"].append((-1, j - 0.5, self.d - 0.5))  # right
        return virtual

    def _make_syndrome_graph(self):
        start_nodes = {"Z": (0.5, 0.5), "X": (0.5, 1.5)}
        for error_key in ["X", "Z"]:
            # subgraphs for each time step
            for t in range(0, self.T):
                start_node = start_nodes[error_key]
                self.S[error_key].add_node(
                    (t,) + start_node,
                    virtual=0,
                    pos=(start_node[1], -start_node[0]),
                    time=t,
                    pos_3D=(
                        start_node[1],
                        -start_node[0],
                        t,
                    ),  # y-coord is flipped for plot purposes
                )
                self.populate_syndrome_graph(
                    (t,) + start_node, t, [], error_key, edge_weight=1
                )

            # connect physical qubits in same location across subgraphs of adjacent times
            syndrome_nodes_t0 = [
                x for x, y in self.S[error_key].nodes(data=True) if y["time"] == 0
            ]
            for node in syndrome_nodes_t0:
                space_label = (node[1], node[2])
                for t in range(0, self.T - 1):
                    self.S[error_key].add_edge(
                        (t,) + space_label, (t + 1,) + space_label, distance=1
                    )

    def _make_syndrome_graph_simulate(self):
        """
        This method injects all possible Pauli errors into the circuit for
        ``code``.
        This is done by examining the qubits used in each gate of the
        circuit for a stored logical 0. A graph is then created with a node
        for each non-trivial syndrome element, and an edge between all such
        elements that can be created by the same error.
        """

        qc = self.code.circuit["0"]

        blank_qc = QuantumCircuit()
        for qreg in qc.qregs:
            blank_qc.add_register(qreg)
        for creg in qc.cregs:
            blank_qc.add_register(creg)

        error_circuit = {}
        circuit_name = {}
        depth = len(qc)
        for j in range(depth):
            qubits = qc.data[j][1]
            for qubit in qubits:
                for error in ["x", "z"]:
                    temp_qc = copy.deepcopy(blank_qc)
                    temp_qc.name = str((j, qubit, error))
                    temp_qc.data = qc.data[0:j]
                    getattr(temp_qc, error)(qubit)
                    temp_qc.data += qc.data[j : depth + 1]
                    circuit_name[(j, qubit, error)] = temp_qc.name
                    error_circuit[temp_qc.name] = temp_qc

        if HAS_AER:
            simulator = Aer.get_backend("qasm_simulator")
        else:
            simulator = BasicAer.get_backend("qasm_simulator")

        job = execute(list(error_circuit.values()), simulator)

        for j in range(depth):
            qubits = qc.data[j][1]
            for qubit in qubits:
                for error in ["x", "z"]:
                    raw_results = {}
                    raw_results["0"] = job.result().get_counts(str((j, qubit, error)))

                    results = self.code.process_results(raw_results["0"])
                    extracted_nodes = self.code.extract_nodes(results)

                    for err_key, nodes in dict(
                        zip(("X", "Z"), extracted_nodes)
                    ).items():
                        # Add virtual syndrome nodes
                        for node in self.virtual[err_key]:
                            # Visualization coords (y-coord flipped for plot)
                            pos_2D = (node[2], -node[1])
                            pos_3D = (*pos_2D, (self.T - 1) / 2)  # Plot midway in stack
                            self.S[err_key].add_node(
                                node, virtual=1, pos=pos_2D, time=-1, pos_3D=pos_3D,
                            )

                        # Add syndrome nodes
                        for node in nodes:
                            # Visualization coords (y-coord flipped for plot)
                            pos_2D = (node[2], -node[1])
                            pos_3D = (*pos_2D, node[0])
                            self.S[err_key].add_node(
                                node,
                                virtual=0,
                                pos=pos_2D,
                                time=node[0],
                                pos_3D=pos_3D,
                            )

                            # Check if any neighbors are virtual nodes
                            candidates = [
                                (-1, node[1] + di, node[2] + dj)
                                for di, dj in product((1, -1), repeat=2)
                            ]
                            for virtual in candidates:
                                if virtual in self.virtual[err_key]:
                                    self.S[err_key].add_edge(node, virtual, distance=1)

                        # Add connections
                        for source, target in combinations(nodes, 2):
                            self.S[err_key].add_edge(source, target, distance=1)

    def populate_syndrome_graph(
        self, current_node, t, visited_nodes, error_key, edge_weight=1
    ):
        """Recursive function to populate syndrome subgraph at time t with error_key X/Z. The current_node
        is connected to neighboring nodes without revisiting a node.
        Args:
            current_node ((t, x, y)): Current syndrome node to be connected with neighboring nodes.
            visited_nodes ([(t, x, y),]): List of syndrome nodes which have already been traver.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
            edge_weight (float, optional): Weight of edge between two adjacent syndrome nodes. Defaults to 1.
        Returns:
            None: function is to traverse the syndrome nodes and connect neighbors
        """
        visited_nodes.append(current_node)
        neighbors = []
        i = current_node[1]  # syndrome node x coordinate
        j = current_node[2]  # syndrome node y coordinate
        neighbors.append((i - 1, j - 1))  # up left
        neighbors.append((i + 1, j - 1))  # down left
        neighbors.append((i - 1, j + 1))  # up right
        neighbors.append((i + 1, j + 1))  # down right

        normal_neighbors = [
            n
            for n in neighbors
            if self.valid_syndrome(n, error_key)
            and (t, n[0], n[1]) not in visited_nodes
        ]  # syndrome node neighbors of current_node not already visited
        virtual_neighbors = [
            n
            for n in neighbors
            if (-1, n[0], n[1]) in self.virtual[error_key]
            and (-1, n[0], n[1]) not in visited_nodes
        ]  # virtual node neighbors of current_node not already visited

        # no neighbors to add edges
        if not normal_neighbors and not virtual_neighbors:
            return

        # add normal/non-virtual neighbors
        for target in normal_neighbors:
            target_node = (
                t,
            ) + target  # target_node has time t with x and y coordinates from target
            if not self.S[error_key].has_node(target_node):
                self.S[error_key].add_node(
                    target_node,
                    virtual=0,
                    pos=(target[1], -target[0]),
                    time=t,
                    pos_3D=(target[1], -target[0], t),
                )  # add target_node to syndrome subgraph if it doesn't already exist
            self.S[error_key].add_edge(
                current_node, target_node, distance=edge_weight
            )  # add edge between current_node and target_node

        # add virtual neighbors
        for target in virtual_neighbors:
            target_node = (
                -1,
            ) + target  # virtual target_node has time -1 with x and y coordinates from target
            if not self.S[error_key].has_node(target_node):
                self.S[error_key].add_node(
                    target_node,
                    virtual=1,
                    pos=(target[1], -target[0]),
                    time=-1,
                    pos_3D=(target[1], -target[0], (self.T - 1) / 2),
                )  # add virtual target_node to syndrome subgraph with z coordinate (T-1)/2 for nice plotting, if it doesn't already exist
            self.S[error_key].add_edge(
                current_node, target_node, distance=edge_weight
            )  # add edge between current_node and virtual target_node

        # recursively traverse normal neighbors
        for target in normal_neighbors:
            self.populate_syndrome_graph(
                (t,) + target, t, visited_nodes, error_key, edge_weight=1
            )

        # recursively traverse virtual neighbors
        for target in virtual_neighbors:
            self.populate_syndrome_graph(
                (-1,) + target, t, visited_nodes, error_key, edge_weight=1
            )

    def valid_syndrome(self, node, error_key):
        """Checks whether a node is a syndrome node under our error_key, which is either X or Z.
        Args:
            node ((t, x, y)): Node in graph.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
        Returns:
            Boolean T/F: whether node is a syndrome node
        """
        i = node[0]
        j = node[1]
        if error_key == "Z":
            if i > 0 and i < self.d - 1 and j < self.d and j > -1:
                return True
            else:
                return False
        elif error_key == "X":
            if j > 0 and j < self.d - 1 and i < self.d and i > -1:
                return True
            else:
                return False

    def make_error_graph(self, nodes, error_key, err_prob=None):
        """Creates error syndrome subgraph from list of syndrome nodes. The output of
        this function is a graph that's ready for minimum weight perfect matching (MWPM).
        If err_prob is specified, we adjust the shortest distance between syndrome
        nodes by the degeneracy of the error path.
        Args:
            nodes ([(t, x, y),]): List of changes of syndrome nodes in time.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
            err_prob (float, optional): Probability of IID data qubit X/Z flip. Defaults to None.
        Returns:
            nx.Graph: Nodes are syndromes, edges are proxy for error probabilities
        """
        paths = {}
        virtual_dict = nx.get_node_attributes(self.S[error_key], "virtual")
        time_dict = nx.get_node_attributes(self.S[error_key], "time")
        error_graph = nx.Graph()
        nodes += self.virtual[error_key]

        for node in nodes:
            if not error_graph.has_node(node):
                error_graph.add_node(
                    node,
                    virtual=virtual_dict[node],
                    pos=(node[2], -node[1]),
                    time=time_dict[node],
                    pos_3D=(node[2], -node[1], time_dict[node]),
                )

        for source, target in combinations(nodes, 2):
            # Distance is proportional to the probability of this error chain, so
            # finding the maximum-weight perfect matching of the whole graph gives
            # the most likely sequence of errors that led to these syndromes.
            distance = int(
                nx.shortest_path_length(
                    self.S[error_key], source, target, weight="distance"
                )
            )

            # If err_prob is specified, we also account for path degeneracies
            deg, path = self._path_degeneracy(source, target, error_key)
            paths[(source, target)] = path
            if err_prob:
                distance = distance - math.log(deg) / (
                    math.log1p(-err_prob) - math.log(err_prob)
                )
            distance = -distance
            error_graph.add_edge(source, target, weight=distance)

        if self.simulation:  # paths incorrect for simulated syndrome graph
            return error_graph

        return error_graph, paths

    def analytic_paths(self, matches, error_key):
        analytic_decoder = GraphDecoder(self.d, self.T)
        paths = {}
        for (source, target) in matches:
            _, path = analytic_decoder._path_degeneracy(
                source[:3], target[:3], error_key
            )
            paths[(source[:3], target[:3])] = path
        return paths

    def _path_degeneracy(self, a, b, error_key):
        """Calculate the number of shortest error paths that link two syndrome nodes
        through both space and time.
        Args:
            a (tuple): Starting or ending syndrome node (degeneracy is symmetric)
            b (tuple): Ending or starting syndrome node (degeneracy is symmetric)
        Raises:
            nx.exception.NodeNotFound: error_key must be X or Z
        Returns:
            int: Number of degenerate shortest paths matching this syndrome pair
            [nodes,]: List of nodes for one of the shortest paths
        """
        # Check which subgraph node is on. If x + y is even => X, else Z.
        # a_sum, b_sum = a[1] + a[2], b[1] + b[2]
        if error_key == "X":
            subgraph = self.S["X"]
        elif error_key == "Z":
            subgraph = self.S["Z"]
        else:
            raise nx.exception.NodeNotFound("error_key must be X or Z")

        shortest_paths = list(nx.all_shortest_paths(subgraph, a, b, weight="distance"))
        one_path = shortest_paths[
            0
        ]  # We can pick any path to return as the error chain
        degeneracy = len(shortest_paths)

        # If either node is a virtual node, we also find degeneracies from the other
        # node to *any* nearest virtual node
        source = None
        if a[0] == -1:
            target = a
            source = b
        elif b[0] == -1:
            target = b
            source = a

        # Compute additional degeneracies to edge boundaries
        if source:
            virtual_nodes = self.virtual[error_key]
            shortest_distance = nx.shortest_path_length(
                subgraph, a, b, weight="distance"
            )
            for node in virtual_nodes:
                distance = nx.shortest_path_length(
                    subgraph, source, node, weight="distance"
                )
                if distance == shortest_distance and node != target:
                    degeneracy += len(
                        list(
                            nx.all_shortest_paths(
                                subgraph, source, node, weight="distance"
                            )
                        )
                    )
        return degeneracy, one_path

    def matching_graph(self, error_graph, error_key):
        """Return subgraph of error graph to be matched.
        Args:
            error_graph (nx.Graph): Complete error graph to be matched.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
        Returns:
            nx.Graph: Subgraph of error graph to be matched
        """
        time_dict = nx.get_node_attributes(self.S[error_key], "time")
        subgraph = nx.Graph()
        syndrome_nodes = [
            x for x, y in error_graph.nodes(data=True) if y["virtual"] == 0
        ]
        virtual_nodes = [
            x for x, y in error_graph.nodes(data=True) if y["virtual"] == 1
        ]

        # add and connect each syndrome node to subgraph
        for node in syndrome_nodes:
            if not subgraph.has_node(node):
                subgraph.add_node(
                    node,
                    virtual=0,
                    pos=(node[2], -node[1]),
                    time=time_dict[node],
                    pos_3D=(node[2], -node[1], time_dict[node]),
                )
        for source, target in combinations(syndrome_nodes, 2):
            subgraph.add_edge(
                source, target, weight=error_graph[source][target]["weight"]
            )

        # connect each syndrome node to its closest virtual node in subgraph
        for source in syndrome_nodes:
            potential_virtual = {}
            for target in virtual_nodes:
                potential_virtual[target] = error_graph[source][target]["weight"]
            nearest_virtual = max(potential_virtual, key=potential_virtual.get)
            paired_virtual = (
                nearest_virtual + source
            )  # paired_virtual (virtual, syndrome) allows for the virtual node to be matched more than once
            subgraph.add_node(
                paired_virtual,
                virtual=1,
                pos=(nearest_virtual[2], -nearest_virtual[1]),
                time=-1,
                pos_3D=(nearest_virtual[2], -nearest_virtual[1], -1),
            )  # add paired_virtual to subgraph
            subgraph.add_edge(
                source, paired_virtual, weight=potential_virtual[nearest_virtual]
            )  # add (syndrome, paired_virtual) edge to subgraph

        paired_virtual_nodes = [
            x for x, y in subgraph.nodes(data=True) if y["virtual"] == 1
        ]

        # add 0 weight between paired virtual nodes
        for source, target in combinations(paired_virtual_nodes, 2):
            subgraph.add_edge(source, target, weight=0)

        return subgraph

    def matching(self, matching_graph, error_key):
        """Return matches of minimum weight perfect matching (MWPM) on matching_graph.
        Args:
            matching_graph (nx.Graph): Graph to run MWPM.
            error_key (char): Which X/Z syndrome subgraph these nodes are from.
        Returns:
            [(node, node),]: List of matchings found from MWPM
        """
        matches = nx.max_weight_matching(matching_graph, maxcardinality=True)
        filtered_matches = [
            (source, target)
            for (source, target) in matches
            if not (len(source) > 3 and len(target) > 3)
        ]  # remove 0 weighted matched edges between virtual syndrome nodes
        return filtered_matches

    def calculate_qubit_flips(self, matches, paths, error_key):
        physical_qubit_flips = {}
        for (source, target) in matches:
            # Trim "paired virtual" nodes to nearest virtual node
            if len(source) > 3:
                source = source[:3]
            if len(target) > 3:
                target = target[:3]

            # Paths dict is encoded in one direction, check other if not found
            if (source, target) not in paths:
                source, target = (target, source)

            path = paths[(source, target)]  # This is an arbitrary shortest error path
            for i in range(0, len(path) - 1):
                start = path[i]
                end = path[i + 1]
                # Check if syndromes are in different physical locations
                # If they're in the same location, this is a measurement error
                if start[1:] != end[1:]:
                    time = start[0]
                    if time == -1:  # Grab time from non-virtual syndrome
                        time = end[0]
                    physical_qubit = (
                        time,
                        (start[1] + end[1]) / 2,
                        (start[2] + end[2]) / 2,
                    )

                    # Paired flips at the same time can be ignored
                    if physical_qubit in physical_qubit_flips:
                        physical_qubit_flips[physical_qubit] = (
                            physical_qubit_flips[physical_qubit] + 1
                        ) % 2
                    else:
                        physical_qubit_flips[physical_qubit] = 1

        physical_qubit_flips = {
            x: error_key for x, y in physical_qubit_flips.items() if y == 1
        }
        return physical_qubit_flips

    def net_qubit_flips(self, flips_x, flips_z):
        flipsx = {flip: "X" for flip, _ in flips_x.items() if flip not in flips_z}
        flipsz = {flip: "Z" for flip, _ in flips_z.items() if flip not in flips_x}
        flipsy = {flip: "Y" for flip, _ in flips_x.items() if flip in flips_z}
        flips = {**flipsx, **flipsy, **flipsz}

        individual_flips = defaultdict(dict)

        for flip, error_key in flips.items():
            individual_flips[flip[1:]][flip[0]] = error_key

        paulis = {
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
            "I": np.array([[1, 0], [0, 1]]),
        }

        physical_qubit_flips = {}
        for qubit_loc, flip_record in individual_flips.items():
            net_error = paulis["I"]
            # print("Physical Qubit: " + str(qubit_loc))
            for time, error in sorted(flip_record.items(), key=lambda item: item[0]):
                # print("Error: " + error + " at time: " + str(time))
                net_error = net_error.dot(paulis[error])
            physical_qubit_flips[qubit_loc] = net_error

        physical_qubit_flips = {
            x: y
            for x, y in physical_qubit_flips.items()
            if not np.array_equal(y, paulis["I"])
        }
        return physical_qubit_flips

    def graph_2D(self, G, edge_label):
        pos = nx.get_node_attributes(G, "pos")
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, edge_label)
        labels = {x: round(y, 3) for (x, y) in labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    def graph_3D(self, G, edge_label, angle=[-116, 22]):
        """Plots a graph with edge labels in 3D.
        Args:
            G (nx.Graph): Graph to plot in 3D.
            edge_label (float): Edge label to display; either distance or weight.
            angle ([float, float]): Initial 3D angle view. Defaults to [-116, 22]
        Returns:
            None: Plot is displayed in plt.show()
        """
        # Get node 3D positions
        pos_3D = nx.get_node_attributes(G, "pos_3D")

        # Define color range based on time
        colors = {
            x: plt.cm.plasma((y["time"] + 1) / self.T) for x, y in G.nodes(data=True)
        }

        # 3D network plot
        with plt.style.context(("ggplot")):

            fig = plt.figure(figsize=(20, 14))
            ax = Axes3D(fig)

            # Loop on the nodes and look up in pos dictionary to extract the x,y,z coordinates of each node
            for node in G.nodes():
                xi, yi, zi = pos_3D[node]

                # Scatter plot
                ax.scatter(
                    xi,
                    yi,
                    zi,
                    color=colors[node],
                    s=120 * (1 + G.degree(node)),
                    edgecolors="k",
                    alpha=0.7,
                )

                # Label node position
                ax.text(xi, yi, zi, node, fontsize=20)

            # Loop on the edges to get the x,y,z, coordinates of the connected nodes
            # Those two points are the extrema of the line to be plotted
            for src, tgt in G.edges():
                x_1, y_1, z_1 = pos_3D[src]
                x_2, y_2, z_2 = pos_3D[tgt]

                x_line = np.array((x_1, x_2))
                y_line = np.array((y_1, y_2))
                z_line = np.array((z_1, z_2))

                # Plot the connecting lines
                ax.plot(x_line, y_line, z_line, color="black", alpha=0.5)

                # Label edges at midpoints
                x_mid = (x_1 + x_2) / 2
                y_mid = (y_1 + y_2) / 2
                z_mid = (z_1 + z_2) / 2
                label = round(G[src][tgt][edge_label], 2)
                ax.text(x_mid, y_mid, z_mid, label, fontsize=14)

        # Set the initial view
        ax.view_init(angle[1], angle[0])

        # Hide the axes
        ax.set_axis_off()

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        plt.show()


# In[6]:


def isNaN(num):
    return num != num


# In[7]:


#import the file with the test dataset here
import pandas as pd
import sys
import matplotlib.pyplot as plt
import copy
import math
from itertools import combinations, product
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_test = pd.read_csv("Classic_decodingDepth3.csv")
"""
def do_decoding(data, depth):
    decoder = GraphDecoder(depth,1)
    G = decoder.S['Z']
    decoder.graph_2D(G,'distance')
    
    df = pd.DataFrame()
    syn = []
    for index, row in data.iterrows():
        d_input = []
        for col in row:
            if not isNaN(col): 
                col = np.array(col.strip('()').split(', ')).astype(float)
                d_input.append(tuple(col))
                
        if index%2 == 0:
            syndromes_x = d_input
            error_graph_x, paths_x = decoder.make_error_graph(syndromes_x,'X')
            matching_graph_x = decoder.matching_graph(error_graph_x,'X')
            matches_x = decoder.matching(matching_graph_x,'X')
            flips_x = decoder.calculate_qubit_flips(matches_x, paths_x,'X')
            syn_x = (translate_errors(flips_x))

        else:
            syndromes_z = d_input
            error_graph_z, paths_z = decoder.make_error_graph(syndromes_z,'Z')
            matching_graph_z = decoder.matching_graph(error_graph_z,'Z')
            matches_z = decoder.matching(matching_graph_z,'Z')
            flips_z = decoder.calculate_qubit_flips(matches_z, paths_z,'Z')
            syn_z = translate_errors(flips_z)
            df = df.append(pd.Series([syn_x, syn_z]), ignore_index=True)   
    return df"""
        


# In[62]:


def add_measurement_errs(curr_syn, prob_err, x_syn, depth):
    #x_syn is True if it is x syndrome, False if it is Z syndrome
    new_syn = []
    if x_syn:
        for i in curr_syn:
            rand = random.random()
            if rand > prob_err:
                new_syn.append(i)
        return (new_syn + return_xmeasurement_errs(depth, prob_err))
    else:
        for i in curr_syn:
            rand = random.random()
            if rand > prob_err:
                new_syn.append(i)
        return (new_syn + return_zmeasurement_errs(depth, prob_err))


# In[63]:


def do_new_decoding(data, depth, prob_err):
    decoder = GraphDecoder(depth,1)
    G = decoder.S['Z']
    decoder.graph_2D(G,'distance')
    
    df = pd.DataFrame()
    syn = []
    for index, row in data.iterrows():
        x_input = []
        z_input = []
        x_type = True
        for col in row:
            if not col == "[]":
                col = eval(col)
                for c in col:
                    if x_type:
                        x_input.append(c)
                    else:
                        z_input.append(c)
            x_type = not x_type  
            
        if prob_err > 0:
            syndromes_x = add_measurement_errs(x_input, prob_err, True, depth)
            syndromes_z = add_measurement_errs(z_input, prob_err, False, depth)
        else:
            syndromes_x = x_input
            syndromes_z = z_input

        error_graph_x, paths_x = decoder.make_error_graph(syndromes_x,'X')
        matching_graph_x = decoder.matching_graph(error_graph_x,'X')
        matches_x = decoder.matching(matching_graph_x,'X')
        flips_x = decoder.calculate_qubit_flips(matches_x, paths_x,'X')
        syn_x = (translate_errors(flips_x))

        error_graph_z, paths_z = decoder.make_error_graph(syndromes_z,'Z')
        matching_graph_z = decoder.matching_graph(error_graph_z,'Z')
        matches_z = decoder.matching(matching_graph_z,'Z')
        flips_z = decoder.calculate_qubit_flips(matches_z, paths_z,'Z')
        syn_z = translate_errors(flips_z)
        df = df.append(pd.Series([syn_x, syn_z]), ignore_index=True)   
    return df


# In[67]:


import random
def return_xmeasurement_errs(depth, prob):
    
    new_errs = []
    
    if depth == 3:
        errs = [(0, -0.5, 0.5), (0, 0.5, 1.5), (0, 1.5, 0.5), (0, 2.5, 1.5)]
    elif depth == 5:
        errs = [(0, -0.5, 0.5), (0, 0.5, 1.5), (0, -0.5, 2.5), (0, 0.5, 3.5), (0, 1.5, 0.5), (0, 1.5, 2.5),
                        (0, 2.5, 1.5), (0, 2.5, 3.5), (0, 3.5, 0.5), (0, 4.5, 1.5), (0, 3.5, 2.5), (0, 4.5, 3.5)]
    else:
        errs = [(0, -0.5, 0.5), (0, 0.5, 1.5), (0, -0.5, 2.5), (0, 0.5, 3.5), (0, -0.5, 4.5), (0, 0.5, 5.5),
                        (0, 1.5, 0.5), (0, 1.5, 2.5), (0, 1.5, 4.5), (0, 2.5, 1.5), (0, 2.5, 3.5), (0, 2.5, 5.5),
                        (0, 3.5, 0.5),  (0, 3.5, 2.5), (0, 3.5, 4.5), (0, 4.5, 1.5), (0, 4.5, 3.5), (0, 4.5, 5.5),
                       (0, 5.5, 0.5), (0, 6.5, 1.5), (0, 5.5, 2.5), (0, 6.5, 3.5), (0, 5.5, 4.5), (0, 6.5, 5.5)]
    for e in errs:
        rand = random.random()
        if rand <= prob:
            new_errs.append(e)
            
    return new_errs
            

def return_zmeasurement_errs(depth, prob):
    
    new_errs = []
    
    if depth == 3:
        errs = [(0, 0.5, 0.5), (0, 0.5, 2.5), (0, 1.5, -0.5), (0, 1.5, 1.5)]
    elif depth == 5:
        errs = [(0, 0.5, 0.5), (0, 0.5, 2.5), (0, 0.5, 4.5), (0, 1.5, -0.5), (0, 1.5, 1.5), (0, 1.5, 3.5),
                        (0, 2.5, 0.5), (0, 2.5, 2.5), (0, 2.5, 4.5), (0, 3.5, -0.5), (0, 3.5, 1.5), (0, 3.5, 3.5)]
    else:
        errs = [(0, 0.5, 0.5), (0, 0.5, 2.5), (0, 0.5, 4.5), (0, 0.5, 6.5), (0, 1.5, -0.5), (0, 1.5, 1.5),
                        (0, 1.5, 3.5), (0, 1.5, 5.5), (0, 2.5, 0.5), (0, 2.5, 2.5), (0, 2.5, 4.5), (0, 2.5, 6.5),
                        (0, 3.5, -0.5),  (0, 3.5, 1.5), (0, 3.5, 3.5), (0, 3.5, 5.5), (0, 4.5, 0.5), (0, 4.5, 2.5),
                       (0, 4.5, 4.5), (0, 4.5, 6.5), (0, 5.5, -0.5), (0, 5.5, 1.5), (0, 5.5, 3.5), (0, 5.5, 5.5)]
        
    for e in errs:
        rand = random.random()
        if rand <= prob:
            new_errs.append(e)
            
    return new_errs


# In[9]:


def translate_errors (phys_errs):
    flipX = np.array([(0, 1),(1, 0)])
    flipZ = np.array([(1, 0), (0, -1)])
    errs = []
    str2 = ""
    for qubit, flip in phys_errs.items():
        row = int(qubit[1])
        col = int(qubit[2])
        if str(flip) == "X":
            str1 = "X"
        elif str(flip) == "Z":
            str1 = "Z"
        else:
            str1 = "X"
            str2 = "Z"
        str1 += str(row) + str(col)
        errs.append(str1)
        if str2 != "":
            str2 += str(row) +str(col)
            errs.append(str2)
            str2 = ""
    return errs            


# In[46]:


decoder = GraphDecoder(3,1)
G = decoder.S['Z']
decoder.graph_2D(G,'distance')


syndromes_x = [(0,.5,1.5)]
error_graph_x, paths_x = decoder.make_error_graph(syndromes_x,'X')
matching_graph_x = decoder.matching_graph(error_graph_x,'X')
matches_x = decoder.matching(matching_graph_x,'X')
flips_x = decoder.calculate_qubit_flips(matches_x, paths_x,'X')

syndromes_z = []
error_graph_z, paths_z = decoder.make_error_graph(syndromes_z,'Z')
matching_graph_z = decoder.matching_graph(error_graph_z,'Z')
matches_z = decoder.matching(matching_graph_z,'Z')
flips_z = decoder.calculate_qubit_flips(matches_z, paths_z,'Z')

flips = decoder.net_qubit_flips(flips_x, flips_z)

print("Result\n")
for qubit, flip in flips.items():
    print("Physical Qubit: " + str(qubit) + "\nFlip: " + "\n" + str(flip) +"\n")


# In[11]:


import math
#extra functions needed
def isNaN(num):
    return num != num

def create_list_from_string(err_list):
    if type(err_list) == float:
        return set([])
    else:
        newstring = err_list.replace("'", "")
        new_err_list = newstring.strip('][').split(', ')
        return set(new_err_list)

def partial_accuracy(y_pred, y_true):
    total = 0
    rows = y_pred.shape[0]
    cols = y_pred.shape[1]
    for i in range(0, rows):
        row_correct = 0
        for j in range(0, cols):
            if y_pred[i,j] == y_true[i,j]:
                row_correct += 1
        total += row_correct/cols
    return total/rows
    


# In[102]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score,accuracy_score, f1_score, recall_score
import time
testData_d3 = pd.read_csv("Graph_depth3_all_combos.csv")
x_test_d3 = testData_d3.drop(['Labels'], axis=1)
y_test_d3 = testData_d3["Labels"].dropna()
y_test_d3 = y_test_d3.apply(lambda x: create_list_from_string(x))

start = time.time()
decoding_d3 = do_new_decoding(x_test_d3, 3, 0)
end = time.time()
print("Time: " + str(end - start))

decoding_d3['combine'] = decoding_d3[[0, 1]].values.tolist()
decoding_d3['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d3 = np.array(decoding_d3[0])

one_hot_d3 = MultiLabelBinarizer()

Y_test_d3 = one_hot_d3.fit_transform(y_test_d3)
pred_d3 = one_hot_d3.transform(decoding_d3)

precision_d3 = precision_score(Y_test_d3, pred_d3, average='micro')
recall_d3 = recall_score(Y_test_d3, pred_d3, average='micro')
f1_d3 = f1_score(Y_test_d3, pred_d3, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d3, recall_d3, f1_d3))
print("Accuracy = ",accuracy_score(Y_test_d3, pred_d3))
print("Accuracy = ",partial_accuracy(Y_test_d3, pred_d3))
print("\n")


# In[86]:


from sklearn.model_selection import train_test_split
testData_d7 = pd.read_csv("Graph_depth7_all_combos.csv")
x_d7 = testData_d7.drop(['Labels'], axis=1)
y_d7 = testData_d7["Labels"].dropna()
y_d7 = y_d7.apply(lambda x: create_list_from_string(x))
#x_test_d7 = x_test_d7.sample(n=5000, random_state=1)
x_train_d7, x_test_d7, Y_train_d7, y_test_d7 = train_test_split(x_d7, y_d7, test_size = .1, shuffle=True)

start = time.time()
decoding_d7 = do_new_decoding(x_test_d7, 7, 0)
end = time.time()
print("Time: " + str(end - start))

decoding_d7['combine'] = decoding_d7[[0, 1]].values.tolist()
decoding_d7['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d7 = np.array(decoding_d7[0])

one_hot_d7 = MultiLabelBinarizer()

Y_test_d7 = one_hot_d7.fit_transform(y_test_d7)
pred_d7 = one_hot_d7.transform(decoding_d7)

precision_d7 = precision_score(Y_test_d7, pred_d7, average='micro')
recall_d7 = recall_score(Y_test_d7, pred_d7, average='micro')
f1_d7 = f1_score(Y_test_d7, pred_d7, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d7, recall_d7, f1_d7))
print("Accuracy = ",accuracy_score(Y_test_d7, pred_d7))
print("Accuracy = ",partial_accuracy(Y_test_d7, pred_d7))
print("\n")


# In[87]:


testData = pd.read_csv("Graph_depth5_all_combos.csv")
x_test = testData.drop(['Labels'], axis=1)
y_test = testData["Labels"].dropna()
y_test = y_test.apply(lambda x: create_list_from_string(x))


# In[88]:


start = time.time()
decoding = do_new_decoding(x_test, 5, 0)
end = time.time()
print("Time: " + str(end - start))


# In[89]:


print(decoding)
decoding['combine'] = decoding[[0, 1]].values.tolist()
decoding['combine'].apply(lambda x: x[0].extend(x[1]))
decoding = np.array(decoding[0])
print(decoding)


# In[90]:


from sklearn.preprocessing import MultiLabelBinarizer
one_hot = MultiLabelBinarizer()

# One-hot encode data for depth of 5
Y_test = one_hot.fit_transform(y_test)
pred = one_hot.transform(decoding)


# In[91]:


print(one_hot.classes_)


# In[92]:


# predict
from sklearn.metrics import precision_score,accuracy_score, f1_score, recall_score, hamming_loss

precision = precision_score(Y_test, pred, average='micro')
recall = recall_score(Y_test, pred, average='micro')
f1 = f1_score(Y_test, pred, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
print("Accuracy = ",accuracy_score(Y_test, pred))
print("Accuracy = ",partial_accuracy(Y_test, pred))
print("\n")


# In[93]:


import time
start = time.time()
decoding_d3 = do_new_decoding(x_test_d3, 3, 0.01)
end = time.time()
print("Time: " + str(end - start))

decoding_d3['combine'] = decoding_d3[[0, 1]].values.tolist()
decoding_d3['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d3 = np.array(decoding_d3[0])

one_hot_d3 = MultiLabelBinarizer()

Y_test_d3 = one_hot_d3.fit_transform(y_test_d3)
pred_d3 = one_hot_d3.transform(decoding_d3)

precision_d3 = precision_score(Y_test_d3, pred_d3, average='micro')
recall_d3 = recall_score(Y_test_d3, pred_d3, average='micro')
f1_d3 = f1_score(Y_test_d3, pred_d3, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d3, recall_d3, f1_d3))
print("Accuracy = ",accuracy_score(Y_test_d3, pred_d3))
print("Accuracy = ",partial_accuracy(Y_test_d3, pred_d3))
print("\n")


# In[94]:


start = time.time()
decoding_d5 = do_new_decoding(x_test, 5, 0.01)
end = time.time()
print("Time: " + str(end - start))

decoding_d5['combine'] = decoding_d5[[0, 1]].values.tolist()
decoding_d5['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d5 = np.array(decoding_d5[0])

one_hot_d5 = MultiLabelBinarizer()

Y_test_d5 = one_hot_d5.fit_transform(y_test)
pred_d5 = one_hot_d5.transform(decoding_d5)

precision_d5 = precision_score(Y_test_d5, pred_d5, average='micro')
recall_d5 = recall_score(Y_test_d5, pred_d5, average='micro')
f1_d5 = f1_score(Y_test_d5, pred_d5, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d5, recall_d5, f1_d5))
print("Accuracy = ",accuracy_score(Y_test_d5, pred_d5))
print("Accuracy = ",partial_accuracy(Y_test_d5, pred_d5))
print("\n")


# In[95]:


start = time.time()
decoding_d7 = do_new_decoding(x_test_d7, 7, 0.01)
end = time.time()
print("Time: " + str(end - start))

decoding_d7['combine'] = decoding_d7[[0, 1]].values.tolist()
decoding_d7['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d7 = np.array(decoding_d7[0])

one_hot_d7 = MultiLabelBinarizer()

Y_test_d7 = one_hot_d7.fit_transform(y_test_d7)
pred_d7 = one_hot_d7.transform(decoding_d7)

precision_d7 = precision_score(Y_test_d7, pred_d7, average='micro')
recall_d7 = recall_score(Y_test_d7, pred_d7, average='micro')
f1_d7 = f1_score(Y_test_d7, pred_d7, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d7, recall_d7, f1_d7))
print("Accuracy = ",accuracy_score(Y_test_d7, pred_d7))
print("Accuracy = ",partial_accuracy(Y_test_d7, pred_d7))
print("\n")


# In[ ]:


#################### MEASUREMENT ERRORS = .03 ##########################


# In[96]:


start = time.time()
decoding_d3 = do_new_decoding(x_test_d3, 3, 0.03)
end = time.time()
print("Time: " + str(end - start))

decoding_d3['combine'] = decoding_d3[[0, 1]].values.tolist()
decoding_d3['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d3 = np.array(decoding_d3[0])

one_hot_d3 = MultiLabelBinarizer()

Y_test_d3 = one_hot_d3.fit_transform(y_test_d3)
pred_d3 = one_hot_d3.transform(decoding_d3)

precision_d3 = precision_score(Y_test_d3, pred_d3, average='micro')
recall_d3 = recall_score(Y_test_d3, pred_d3, average='micro')
f1_d3 = f1_score(Y_test_d3, pred_d3, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d3, recall_d3, f1_d3))
print("Accuracy = ",accuracy_score(Y_test_d3, pred_d3))
print("Accuracy = ",partial_accuracy(Y_test_d3, pred_d3))
print("\n")


# In[97]:


start = time.time()
decoding_d5 = do_new_decoding(x_test, 5, 0.03)
end = time.time()
print("Time: " + str(end - start))

decoding_d5['combine'] = decoding_d5[[0, 1]].values.tolist()
decoding_d5['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d5 = np.array(decoding_d5[0])

one_hot_d5 = MultiLabelBinarizer()

Y_test_d5 = one_hot_d5.fit_transform(y_test)
pred_d5 = one_hot_d5.transform(decoding_d5)

precision_d5 = precision_score(Y_test_d5, pred_d5, average='micro')
recall_d5 = recall_score(Y_test_d5, pred_d5, average='micro')
f1_d5 = f1_score(Y_test_d5, pred_d5, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d5, recall_d5, f1_d5))
print("Accuracy = ",accuracy_score(Y_test_d5, pred_d5))
print("Accuracy = ",partial_accuracy(Y_test_d5, pred_d5))
print("\n")


# In[98]:


start = time.time()
decoding_d7 = do_new_decoding(x_test_d7, 7, 0.03)
end = time.time()
print("Time: " + str(end - start))

decoding_d7['combine'] = decoding_d7[[0, 1]].values.tolist()
decoding_d7['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d7 = np.array(decoding_d7[0])

one_hot_d7 = MultiLabelBinarizer()

Y_test_d7 = one_hot_d7.fit_transform(y_test_d7)
pred_d7 = one_hot_d7.transform(decoding_d7)

precision_d7 = precision_score(Y_test_d7, pred_d7, average='micro')
recall_d7 = recall_score(Y_test_d7, pred_d7, average='micro')
f1_d7 = f1_score(Y_test_d7, pred_d7, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d7, recall_d7, f1_d7))
print("Accuracy = ",accuracy_score(Y_test_d7, pred_d7))
print("Accuracy = ",partial_accuracy(Y_test_d7, pred_d7))
print("\n")


# In[ ]:


#################### MEASUREMENT ERRORS: .05 #############################


# In[99]:


start = time.time()
decoding_d3 = do_new_decoding(x_test_d3, 3, 0.05)
end = time.time()
print("Time: " + str(end - start))

decoding_d3['combine'] = decoding_d3[[0, 1]].values.tolist()
decoding_d3['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d3 = np.array(decoding_d3[0])

one_hot_d3 = MultiLabelBinarizer()

Y_test_d3 = one_hot_d3.fit_transform(y_test_d3)
pred_d3 = one_hot_d3.transform(decoding_d3)

precision_d3 = precision_score(Y_test_d3, pred_d3, average='micro')
recall_d3 = recall_score(Y_test_d3, pred_d3, average='micro')
f1_d3 = f1_score(Y_test_d3, pred_d3, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d3, recall_d3, f1_d3))
print("Accuracy = ",accuracy_score(Y_test_d3, pred_d3))
print("Accuracy = ",partial_accuracy(Y_test_d3, pred_d3))
print("\n")


# In[100]:


start = time.time()
decoding_d5 = do_new_decoding(x_test, 5, 0.05)
end = time.time()
print("Time: " + str(end - start))

decoding_d5['combine'] = decoding_d5[[0, 1]].values.tolist()
decoding_d5['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d5 = np.array(decoding_d5[0])

one_hot_d5 = MultiLabelBinarizer()

Y_test_d5 = one_hot_d5.fit_transform(y_test)
pred_d5 = one_hot_d5.transform(decoding_d5)

precision_d5 = precision_score(Y_test_d5, pred_d5, average='micro')
recall_d5 = recall_score(Y_test_d5, pred_d5, average='micro')
f1_d5 = f1_score(Y_test_d5, pred_d5, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d5, recall_d5, f1_d5))
print("Accuracy = ",accuracy_score(Y_test_d5, pred_d5))
print("Accuracy = ",partial_accuracy(Y_test_d5, pred_d5))
print("\n")


# In[101]:


start = time.time()
decoding_d7 = do_new_decoding(x_test_d7, 7, 0.05)
end = time.time()
print("Time: " + str(end - start))

decoding_d7['combine'] = decoding_d7[[0, 1]].values.tolist()
decoding_d7['combine'].apply(lambda x: x[0].extend(x[1]))
decoding_d7 = np.array(decoding_d7[0])

one_hot_d7 = MultiLabelBinarizer()

Y_test_d7 = one_hot_d7.fit_transform(y_test_d7)
pred_d7 = one_hot_d7.transform(decoding_d7)

precision_d7 = precision_score(Y_test_d7, pred_d7, average='micro')
recall_d7 = recall_score(Y_test_d7, pred_d7, average='micro')
f1_d7 = f1_score(Y_test_d7, pred_d7, average='micro')
   
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_d7, recall_d7, f1_d7))
print("Accuracy = ",accuracy_score(Y_test_d7, pred_d7))
print("Accuracy = ",partial_accuracy(Y_test_d7, pred_d7))
print("\n")


# In[103]:


print(len(x_test_d3))
print(len(x_test))
print(len(x_test_d7))

