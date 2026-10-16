from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .geometry import Edge, Face, canonical_edge
from .mesh import adjacency_from_faces, edge_to_faces


def connected_components(adjacency: Sequence[Set[int]]) -> int:
    unseen = set(range(len(adjacency)))
    count = 0
    while unseen:
        count += 1
        root = unseen.pop()
        queue = [root]
        while queue:
            vertex = queue.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return count


def boundary_loops(faces: Sequence[Face]) -> int:
    edge_owners = edge_to_faces(faces)
    boundary_edges = [edge for edge, owners in edge_owners.items() if len(owners) == 1]
    if not boundary_edges:
        return 0

    boundary_adjacency: Dict[int, Set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        boundary_adjacency[a].add(b)
        boundary_adjacency[b].add(a)

    unseen = set(boundary_adjacency)
    components = 0
    while unseen:
        components += 1
        root = unseen.pop()
        queue = [root]
        while queue:
            vertex = queue.pop()
            for neighbor in boundary_adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return components


def orientability_check(faces: Sequence[Face]) -> bool:
    directed_edges: List[Dict[Edge, int]] = []
    edge_owners = edge_to_faces(faces)

    for face in faces:
        a, b, c = face
        local: Dict[Edge, int] = {}
        for source, target in ((a, b), (b, c), (c, a)):
            edge = canonical_edge(source, target)
            local[edge] = 1 if (source, target) == edge else -1
        directed_edges.append(local)

    relation_graph: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for edge, owners in edge_owners.items():
        if len(owners) != 2:
            continue
        face_a, face_b = owners
        direction_a = directed_edges[face_a][edge]
        direction_b = directed_edges[face_b][edge]
        relation = -direction_a * direction_b
        relation_graph[face_a].append((face_b, relation))
        relation_graph[face_b].append((face_a, relation))

    signs: Dict[int, int] = {}
    for root in range(len(faces)):
        if root in signs:
            continue
        signs[root] = 1
        queue = deque([root])
        while queue:
            face = queue.popleft()
            for neighbor, relation in relation_graph[face]:
                expected = signs[face] * relation
                if neighbor in signs and signs[neighbor] != expected:
                    return False
                if neighbor not in signs:
                    signs[neighbor] = expected
                    queue.append(neighbor)
    return True


@dataclass(frozen=True)
class Invariants:
    vertices: int
    edges: int
    faces: int
    euler_characteristic: int
    connected_components: int
    boundary_loops: int
    nonmanifold_edges: int
    orientable: bool


def compute_invariants(n_vertices: int, faces: Sequence[Face]) -> Invariants:
    edge_owners = edge_to_faces(faces)
    edge_count = len(edge_owners)
    adjacency = adjacency_from_faces(n_vertices, faces)
    return Invariants(
        vertices=n_vertices,
        edges=edge_count,
        faces=len(faces),
        euler_characteristic=n_vertices - edge_count + len(faces),
        connected_components=connected_components(adjacency),
        boundary_loops=boundary_loops(faces),
        nonmanifold_edges=sum(len(owners) > 2 for owners in edge_owners.values()),
        orientable=orientability_check(faces),
    )