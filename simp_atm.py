from itertools import combinations, permutations
from collections import deque, defaultdict
from collections.abc import Mapping, Iterator, Callable, Hashable, Sized, Set as BaseSet

# -------------- Classes --------------

class ViewDict[K, V](Mapping[K, V]):
    """
    A class that implements a view-only dictionary given an underlying
    dictionary and a set of keys. Useful to avoid creating a new dictionary
    every time one needs to work with a subdictionary.
    """

    def __init__(self, base_dict: Mapping[K, V], key_subset: BaseSet[K], *, assume_subset: bool = False) -> None:
        self._base_dict = base_dict
        self._subset = frozenset(key_subset)
        self._assume_subset = assume_subset

    def __getitem__(self, key: K) -> V:
        if key in self._subset and (self._assume_subset or key in self._base_dict):
            return self._base_dict[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        return iter(self._subset if self._assume_subset else self._subset & self._base_dict.keys())

    def __len__(self) -> int:
        return len(self._subset if self._assume_subset else self._subset & self._base_dict.keys())

Mask = int
Edge = tuple[int, int]
OGPos = dict[Edge, int] | ViewDict[Edge, int]

# -------- Low level utilities --------

def bits_set(bitmask: Mask) -> Iterator[int]:
    """
    Yield the set bit positions (indices) of a bitmask in ascending order.
    """
    
    mask = bitmask
    while mask:
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        yield i
        mask ^= lsb

def generate_bitmasks(k: int, m: int) -> Iterator[Mask]:
    """
    Generate all bitmasks with exactly k bits set from the indices {0..m-1}.
    """

    for comb in combinations(range(m), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        yield mask

def relabel_mask(mask: Mask, relabeling: dict[int, int]) -> Mask:
    """
    Apply a relabeling to a bitmask. relabeling[i] = j means element i moves
    to position j. In the output mask, bit relabeling[i] is set if bit i was
    set in the original mask.
    """
    new_mask = 0
    for i in bits_set(mask):
        try:
            new_mask |= (1 << relabeling[i])
        except KeyError:
            raise KeyError("The relabeling has insufficient information.")
    
    return new_mask

def find_diff_index(mask1: Mask, mask2: Mask) -> int:
    """
    Given two bitmasks return the index where they differ if they only differ
    at one bit, and otherwise raise a ValueError.
    """
    diff = mask1 & ~mask2 

    # Validate: should have exactly one bit set
    if diff == 0 or (diff & (diff - 1)) != 0:
        raise ValueError("Masks don't differ at exactly one position")
    
    # Find the index of the single set bit
    return diff.bit_length() - 1

def tuple_to_mask(tup: tuple[int]) -> Mask:
    """
    Convert a tuple of vertex indices to a bitmask.
    """
    mask = 0
    for v in tup:
        mask |= (1 << v)
    return mask

def mask_to_tuple(mask: Mask) -> tuple[int]:
    """
    Convert a bitmask to a sorted tuple of bit indices.
    """
    return tuple(bits_set(mask))

def mask_to_string(mask: Mask) -> string:
    """
    Convert a bitmask to a string of its sorted bit indices.
    """
    return ''.join(str(i) for i in bits_set(mask))

def bitmask_average(mask: Mask) -> int:
    """
    Calculate the average of the indices that are set in a bitmask.
    """
    weighted_sum = 0
    temp = mask
    i = 0
    while temp:
        weighted_sum += i * (temp & 1)
        temp >>= 1
        i += 1
    
    return weighted_sum / mask.bit_count()

def generate_sublists(n: int):
    """
    Generate all nonempty sublists of [1, 2, ..., n].
    """
    for k in range(1, n + 1):
        yield from combinations(range(1, n + 1), k)

# --------- Search utilities ----------

def validated_set(edge: Edge, sign: int, ogpos: OGPos, queue) -> bool:
    """
    Set ogpos[edge] = sign if it is not set, and check consistency agains
    provided sign. If it was set enqueue it for propagation.
    """
    if edge in ogpos:
        return ogpos[edge] == sign

    ogpos[edge] = sign
    queue.append(edge)
    return True

def enforce_oriented_thinness(a: Edge, b: Edge, c: Edge, d: Edge, ogpos: OGPos, queue) -> bool:
    """
    Enforce oriented thinness on diamond with edges a,b,c, and d:
    ogpos[a]*ogpos[b]*ogpos[c]*ogpos[d] = -1. If all four signs are known,
    return True if the condition is satisfied. If only three are known imply
    and try to set the fourth one. Otherwise do nothing and return True.
    """
    
    known_signs = []
    unknown_edges = []
    for edge in (a, b, c, d):
        if edge in ogpos:
            known_signs.append(ogpos[edge])
        else:
            unknown_edges.append(edge)
    
    if len(unknown_edges) == 0:
        prod = 1
        for v in known_signs:
            prod *= v
        return prod == -1
    elif len(unknown_edges) == 1:
        prod_known = 1
        for v in known_signs:
            prod_known *= v

        if not validated_set(unknown_edges[0], -1 * prod_known, ogpos, queue):
            return False
    
    return True

def propagate_from_edge(edge: Edge, ogpos: OGPos, queue) -> bool:
    """
    Bottom-up propagation from vertices with low bit_count to vertices with
    higher bit_count. We look at diamonds whose top is s, given the edge
    (e, s). We only enforce oriented thinness for diamonds whose top element
    has bit_count greater than 2, to avoid appending a minimum to ogpos.
    """

    e, s = edge
    if s.bit_count() >= 3:
        #s with bit e unset
        s_prime = s & ~(1 << e)

        for x in bits_set(s_prime):
            # Diamond edges are: (e, s), (x, s), (e, s - {x}), (x, s - {e})
            a = (e, s)
            b = (x, s)
            c = (e, s & ~(1 << x))
            d = (x, s_prime)
            if not enforce_oriented_thinness(a, b, c, d, ogpos, queue):
                return False

    return True

def propagate(ogpos: OGPos, queue) -> bool:
    """
    Go through the queue, enforcing oriented thinness on diamonds whose tops are in the queue.
    """
    
    while queue:
        edge = queue.popleft()
        if not propagate_from_edge(edge, ogpos, queue):
            return False

    return True

def seed_edges(n: int) -> OGPos:
    """
    To break the S_n symmetry we manually seed the edges connecting a
    1-simplex to a 0-simplex by choosing: ogpos[(i, {i,j})] = +1, and
    ogpos[(j, {i,j})] = -1. 
    """

    ogpos = {}
    # this was need for some boundary calculation. maybe fix later?
    for i in range(n + 1):
        s = (1 << i)
        ogpos[(i, s)] = 0

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            s = (1 << i) | (1 << j)
            ogpos[(i, s)] = +1
            ogpos[(j, s)] = -1
    
    return ogpos

def build_edge_order(n: int) -> list[Edge]:
    """
    Returns a list of edges (e, s) ordered in the order in which the search
    should go, namely: first order based on s.bit_count() starting from 3
    (lower edges are already seeded), then order based on the bitmask s itself,
    and finally order based on the bit index e.
    """

    edges = []
    for k in range(3, n + 2):
        for s in generate_bitmasks(k, n + 1):
            for e in bits_set(s):
                edges.append((e, s))

    edges.sort(key=lambda edge: (edge[1].bit_count(), edge[1], edge[0]))
    return edges

def generate_oriented_thin(n: int) -> Iterator[OGPos]:
    """
    Enumerate all oriented thin, oriented graded poset structures on the standard n-simplex.
    """

    if n < 0:
        raise ValueError("{} is a negative integer.".format(n))

    
    ogpos_init = seed_edges(n)
    edges = build_edge_order(n)

    def dfs(i: int, ogpos: OGPos) -> Iterator[OGPos]:

        while i < len(edges) and edges[i] in ogpos:
            i += 1

        if i == len(edges):
            # yield and cut off this branch
            yield ogpos
            return

        edge = edges[i]
        for sign in (+1, -1):
            
            # efficiency could be improved here if necessary
            local_copy = ogpos.copy()

            queue = deque()
            if not validated_set(edge, sign, local_copy, queue):
                continue
            
            if not propagate(local_copy, queue):
                continue
            
            yield from dfs(i + 1, local_copy)

    yield from dfs(0, ogpos_init)

def atom_candidates(n: int) -> Iterator[OGPos]:
    """
    Yield simplicial atom candidates by generating all oriented-thin, oriented
    graded poset structures on the standard n-simplex, and then checking that
    they are globular, and that their slices admit a topological order (which
    for frame-acyclic atoms also guarantees the existence of layerings).
    """

    for ogpos in generate_oriented_thin(n):
        if is_globular(n, ogpos):
            if has_layerings(n, ogpos):
                yield ogpos

def orbits_of_flip_action(n: int) -> list[OGPos]:
    orbit_representatives = []

    for atom in atom_candidates(n):
        seen = False
        for indices in generate_sublists(n):
            if not seen:
                flipped_atom = flip_molecule(indices, atom, n)
                if flipped_atom in orbit_representatives:
                    seen = True
        
        if not seen:
            orbit_representatives.append(atom)
    
    return orbit_representatives

#  ----- Combinatorial Operations -----

def recover_total_order(elements: list[Hashable], compare: Callable[[Hashable, Hashable], int], unique: bool = False) -> list[Hashable] | None:
    """
    Recover a total ordering from pairwise comparison oracle.

    Logic:
        We build an directed adjacency graph out of the comparisons and the use
        a version of Kahn's algorithm to walk along a path in the directed
        graph and see if we cover all of it.
    
    Input:
        elements: list of distinct elements to order
        compare:  function f(a, b) returning:
                  +1 if a precedes b in total order
                  -1 if b precedes a in total order
                  0 if a, b are not adjacent in total order
        unique:   boolean that insists on the total order being unique if
                  set to True
    
    Output:
        List of elements in recovered order, or None if either no valid total
        order exists, or unique = True and a total order is underdetermined.
    """

    graph = {a: set() for a in elements}    
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            result = compare(a, b)
            if result == 1:
                graph[a].add(b)
            elif result == -1:
                graph[b].add(a)
            elif result != 0:
                raise ValueError("Oracle returned invalid entry {}.".format(result))
    
    indegree = {a: 0 for a in elements}
    for a in graph:
        for b in graph[a]:
            indegree[b] += 1
    
    queue = deque([v for v in elements if indegree[v] == 0])
    total_order = []
    while queue:
        # Uniqueness condition: exactly one source available at each step
        if unique and len(queue) > 1:
            return None
        
        a = queue.popleft()
        total_order.append(a)
        
        # Decrease in-degrees of neighbors
        for b in graph[a]:
            indegree[b] -= 1
            if indegree[b] == 0:
                queue.append(b)
    
    # Cycle detection: not all vertices visited
    if len(total_order) != len(elements):
        return None
    
    return total_order

#  ----- Mathematical Operations ------

def dimension(ogpos: OGPos) -> int:
    # It's a bit strange to calculate this. Just keep track of it by hand.
    return max(edge[1].bit_count() for edge in ogpos) - 1

def downset(faces: set , ogpos: OGPos) -> OGPos:
    """
    Given a set of faces return the oriented graded subposet that is the
    downward closure of the set of faces. If some of the faces are not actually
    in the oriented graded poset then those are ignored.
    """

    closure_edges = set()
    for face in faces:
        subface = face
        while subface != 0:
            for i in bits_set(subface):
                if (i, subface) in ogpos:
                    closure_edges.add((i, subface))
            subface = (subface - 1) & face
    
    return ViewDict(ogpos, closure_edges, assume_subset=True)

def faces(sign: int, element: Mask, ogpos: OGPos) -> set[Mask]:
    faces_set = set()
    for i in bits_set(element):
        if (i, element) in ogpos:
            if ogpos[(i, element)] == sign:
                faces_set.add(element & ~(1 << i))
    
    return faces_set

def boundary_top_cells(sign: int, ogpos: OGPos, k: int | None = None) -> set[Mask]:
    """
    This function finds the top cells of the boundary.

    Logic:
        The function finds all k-cells that have no (-sign)-cofaces, by looping
        through all Hasse edges, finding those whose top vertex is a (k+1)-cell, 
        and remembering whether the sign of this edge is (-sign). If a bottom vertex
        appears at some point, but never as part of a (-sign)-edge then it is a top
        cell of the boundary in question.

    Input:
        sign in (-1, +1): chooses which boundary to take,
        ogpos: Dict or ViewDict that provides access to signs of Hasse edges
            connecting levels k and k+1 of the orinted gaded poset. Only those
            are necessary to calculate the k-boundary
        k: integer that tells us to calculate the k-th boundary. If not provided
            the default is the (n-1)-boundary where n is the dimension of the
            oriented graded poset.
    
    Output:
        The output is the set of top faces of the (sign, k)-boundary.
    """

    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    if not ogpos:
        return set()
    if k == None:
        print("WARNING: It's dangerous to not provide the boundary dimension explicitly.")
        k = next(reversed(ogpos))[1].bit_length() - 2

    all_faces = set()
    wrong_sign_faces = set()
    for (e, s) in ogpos:
        if s.bit_count() == k + 2:
            t = s ^ (1 << e)
            all_faces.add(t)
            if ogpos[(e, s)] == - sign:
                wrong_sign_faces.add(t)

    correct_sign_faces = all_faces.difference(wrong_sign_faces)

    return correct_sign_faces

def boundary_top_edges(sign: int, ogpos: OGPos, k: int | None = None) -> OGPos:
    """
    This function finds the top cells of the boundary and their immediate downward
    relations, which is a first step in calculating the actual full boundary. In
    many cases it is enough to only know these Hasse edges so this function is separate.

    Logic:
        The function delagates finding the top cells to boundary_top_cells and then
        uses the simplicial property to calculate all their immediate descendants.

    Input:
        sign in (-1, +1): chooses which boundary to take,
        ogpos: Dict or ViewDict that provides access to signs of Hasse edges
            connecting levels k and k+1 of the orinted gaded poset. Only those
            are necessary to calculate the k-boundary
        k: integer that tells us to calculate the k-th boundary. If not provided
            the default is the (n-1)-boundary where n is the dimension of the
            oriented graded poset.
    
    Output:
        The output is a ViewDict whose underlying dictionary is the underlying
        dictionary of ogpos, and whose keys are the Hasse edges whose higher
        vertex is a top face of the k-boundary.
    """

    top_cells = boundary_top_cells(sign, ogpos, k)
    top_Hasse_edges = {(e, s) for s in top_cells for e in bits_set(s)}

    if type(ogpos) is ViewDict:
        boundary_top_ogpos = ViewDict(ogpos._base_dict, top_Hasse_edges, assume_subset=True)
    else: # type(ogpos) is dict
        boundary_top_ogpos = ViewDict(ogpos, top_Hasse_edges, assume_subset=True)

    return boundary_top_ogpos

def boundary(sign: int, ogpos: OGPos, k: int | None = None) -> OGPos:
    """
    This function returns the full (sign, k)-boundary of ogpos. In many cases
    you do not need to calculate the full boundary for further calculations. 

    Logic:
        The function delagates finding the top cells to boundary_top_cells and then
        returns the downward closure of those top cells.

    Input:
        sign in (-1, +1): chooses which boundary to take,
        ogpos: Dict or ViewDict that provides access to signs of Hasse edges
            connecting levels k and k+1 of the orinted gaded poset. Only those
            are necessary to calculate the k-boundary
        k: integer that tells us to calculate the k-th boundary. If not provided
            the default is the (n-1)-boundary where n is the dimension of the
            oriented graded poset.
    
    Output:
        The output is a ViewDict whose underlying dictionary is the underlying
        dictionary of ogpos, and whose keys are the Hasse edges of the full
        (sign, k)-boundary.
    """
    return downset(boundary_top_cells(sign, ogpos, k), ogpos)

def get_top_cells(n: int, ogpos: OGPos) -> set[Mask]:
    return set(s for (_, s) in ogpos if s.bit_count() == n + 1)

def order_of_top_cells(n: int, ogpos: OGPos) -> list[Mask] | None:
    """
    Returns a topological order on the top cells of an n-dimensional oriented
    graded poset, if it exists, otherwise return None.
    """

    def local_order(x, y):

        if x == y:
            return 0
        
        # for round molecule we can guarantee that if the closures of
        # x and y intersect then the maximal element of the intersection
        # is of dimension one less and is the intersection of the bitmasks
        try:
            a = find_diff_index(x, y)
            b = find_diff_index(y, x)
        except ValueError:
            return 0
        
        sgn1 = ogpos[(a, x)]
        sgn2 = ogpos[(b, y)]

        if sgn1 in (-1, +1) and sgn2 in (-1, +1) and sgn1 != sgn2:
            return sgn1
        else:
            return 0
    
    top_cells = list(get_top_cells(n, ogpos))

    return recover_total_order(top_cells, local_order)

def molecular_slices(n: int, ogpos: OGPos, output: string = 'full') -> list[set[Mask]] | list[OGPos] | None:
    """
    Given an oriented graded poset it tries to produce an ordered list of
    slices starting with the negative boundary and performing one atomic
    rewrite at a time until it runs out of rewrites.
    """
    order = order_of_top_cells(n, ogpos)

    if order is None:
        return None
    
    boun_m = boundary_top_cells(-1, ogpos, n - 1)
    slices = [boun_m]
    for x in order:
        new_slice = slices[-1].difference(faces(-1, x, ogpos)).union(faces(+1, x, ogpos))
        slices.append(new_slice)

    if output == 'top':
        return slices
    if output == 'full':
        return [downset(slc, ogpos) for slc in slices]
    else:
        raise ValueError("output can only be 'top' or 'full'.")

def flip_molecule(indices: list[tuple[int,...]], molecule: OGPos, n: int) -> OGPos:
    """
    Given a subset (indices) of [1, 2, ..., n], where n = dimimension(molecule),
    construct a new molecule, which has all k-cells flipped where k is in indices.
    """

    if max(indices) > n:
        raise ValueError("Certain indices in {} are larger than {}.".format(indices, n))

    if 1 in indices:
        perm = {i: n - i for i in range(n + 1)}
        flipped_molecule = {(perm[e], relabel_mask(s, perm)): (-sgn if s.bit_count() - 1 in indices else sgn) for (e, s), sgn in molecule.items()}
    else:
        flipped_molecule = {key: (-sgn if key[1].bit_count() - 1 in indices else sgn) for key, sgn in molecule.items()}
    
    return flipped_molecule

def get_vertices(ogpos: OGPos) -> tuple[int, ...]:
    """
    Given an oriented graded poset ogpos, return a tuple of integers
    representing which vertices are ogpos.
    """

    full_mask = 0
    for (e, s) in ogpos:
        full_mask |= s
    
    return mask_to_tuple(full_mask)

def induced_relabeling(molecule1: OGPos, molecule2: OGPos) -> dict[int, int]:
    """
    Given two molecules return the conversion that converts the indices of the
    vertices of molecule1 to the indices of the vertices of molecule2 in the
    form of a dictionary whose keys are the from the first collection of
    indices and the values are from the second collection. If the number of
    vertices is not the same it raises a ValueError.
    """

    vertices1 = get_vertices(molecule1)
    vertices2 = get_vertices(molecule2)
    
    return dict(zip(vertices1, vertices2, strict=True))

#  ------- Mathematical Checks --------

def is_globular(n: int, ogpos: OGPos) -> bool:

    bdrp = ogpos
    bdrm = ogpos

    for k in range(n - 1, -1, -1):

        bdrpp = boundary_top_edges(+1, bdrp, k)
        bdrmp = boundary_top_edges(+1, bdrm, k)
        if bdrpp != bdrmp:
            return False

        bdrpm = boundary_top_edges(-1, bdrp, k)
        bdrmm = boundary_top_edges(-1, bdrm, k)
        if bdrpm != bdrmm:
            return False

        direct_bdrp = boundary_top_edges(+1, ogpos, k)
        if direct_bdrp != bdrpp:
            return False
        
        direct_bdrm = boundary_top_edges(-1, ogpos, k)
        if direct_bdrm != bdrmm:
            return False
        
        bdrp = bdrpp
        bdrm = bdrmm

    return True

def has_layerings(n: int, ogpos: OGPos) -> bool:
    """
    Tries to recursively generate molecular slices, so that it provides a topological k-ordering of each k-dimensional slice. For round molecules, such an ordering is guaranteed to exist, so we get a nontrivial check on the oriented graded poset being a simplicial atom....
    """
    queue = deque([(ogpos, n)])
    while queue:

        mol, dim = queue.popleft()
        slices = molecular_slices(dim, mol)
        
        if slices is None:
            return False
        
        if dim > 1:
            queue.extend([(slc, dim - 1) for slc in slices])

    return True

def are_isomorphic(molecule1: OGPos, molecule2: OGPos) -> bool:
    """
    Checks if two molecules are isomorphic, by tring to constructing a
    relabeling of the vertices, and then checking if that relabeling actually
    produces the same signs for the Hasse edges. 
    """
    
    try:
        relabeling = induced_relabeling(molecule1, molecule2)
    except ValueError:
        # ValueError is raised if one molecule has more vertices than the other
        return False
    
    print(relabeling)

    for (e, s) in molecule1:
        relabeled_edge = (relabeling[e], relabel_mask(s, relabeling))
        if relabeled_edge not in molecule2:
            return False
        if molecule1[(e,s)] != molecule2[relabeled_edge]:
            return False
    
    return True

#  --------- 4-Atom Specific ----------

def boundary_slices_for_4_atom(atom: OGPos) -> list[OGPos]:
    bm = boundary(-1, atom, 4 - 1)
    bp = boundary(+1, atom, 4 - 1)

    slcm = molecular_slices(3, bm)
    slcp = molecular_slices(3, bp)

    if slcm is None or slcp is None:
        raise ValueError("Your input is not a molecule.")

    return slcm + slcp[-2:0:-1]

def orbits_of_slice_permutation_for_4_atoms() -> list[tuple[int, OGPos]]:
    
    L = 5# = len(slices1) = len(slices2)

    def check_alignment(slices1: list[OGPos], slices2: list[OGPos], shift: int) -> bool:
        for i in range(L):
            #if not are_isomorphic(slices1[i], slices2[(i + shift) % L]):
            if slices1[i] != slices2[(i + shift) % L]:
                return False
            return True

    def check_molecule(molecule: OGPos) -> bool:
        for _, molecule2 in reversed(orbit_representatives):
            slices1 = boundary_slices_for_4_atom(molecule)
            slices2 = boundary_slices_for_4_atom(molecule2)
            
            for shift in range(-1, L - 1):
                if check_alignment(slices1, slices2, shift):
                    return True
            
            for shift in range(-1, L - 1):
                if check_alignment(slices1, slices2[::-1], shift):
                    return True
            
            return False
    
    orbit_representatives = []

    for i, molecule in enumerate(orbits_of_flip_action(4)):
        if not check_molecule(molecule):
            orbit_representatives.append((i, molecule))
    
    return orbit_representatives

#  -------- Tikz Generation -----------

def coordinates_for_2_molecule(molecule: OGPos) -> dict[Mask, list[float]]:
    """
    Given a 2-molecule it produces a dictionary whose keys are the 2-cells and whose values are lists [x1, y1, x2, y2, x3, y3] of the x and y coordinates of the three legs in the string diagram representation of the 2-molecules. The x coordinates are calculated as average of the pairs of integers inside a leg, and the y coordinates are the levels at roughly the levels at which the leg is first encountered...... bit of a mess
    """
    tops = get_top_cells(2, molecule)
    slices_tops = molecular_slices(2, molecule, output='top')

    if slices_tops is None:
        #raise ValueError('...')
        return None

    coords = {}
    for (i, slc) in enumerate(slices_tops):
        for s in slc:
            if s not in coords:
                coords[s] = (bitmask_average(s), [i])
            else:
                lst = coords[s][1]
                lst.append(i)
                coords[s] = (coords[s][0], lst)

    connections = {}
    for x in tops:
        vert_or = 0
        for i in bits_set(x):
            vert_or += molecule[(i, x)]
        if vert_or not in (-1, +1):
            raise ValueError("The input is not a valid 2 molecule.")
        
        for s in coords:
            if s & x == s:
                connections[s] = connections.get(s, []) + [(x, vert_or)]
    
    for s in coords:
        x_coord_s = coords[s][0]
        if len(connections[s]) == 1:
            y_coord_s = (min(coords[s][1]) if connections[s][0][1] == -1 else max(coords[s][1]))
        elif len(connections[s]) == 2:
            if connections[s][0][1] != connections[s][1][1]:
                y_coord_s = (min(coords[s][1]) + max(coords[s][1])) / 2
            else:
                y_coord_s = (min(coords[s][1]) if connections[s][0][1] == -1 else max(coords[s][1]))
        else:
            raise ValueError("Something went wrong. Each line can only touch two simplices.")
        coords[s] = [x_coord_s, y_coord_s]

    trivalent_vertices = {}
    for x in tops:
        for s in coords:
            if s & x == s:
                trivalent_vertices[x] = trivalent_vertices.get(x, []) + coords[s]
    
    return trivalent_vertices

def tikz_code_for_2_molecule(molecule: OGPos) -> string:
    trival = coordinates_for_2_molecule(molecule)

    if trival is None:
        return ''

    code = r"""\begin{tikzpicture}[very thick, color=green!50!black]"""
    for x in trival:
        code += r"""
                \tktrivalcoord{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}""".format(trival[x][0], trival[x][1], trival[x][2], trival[x][3], trival[x][4], trival[x][5])
    code += r"""
            \end{tikzpicture}"""

    return code

def tikz_code_for_4_molecule(i: int, molecule: OGPos, special: int or None = None) -> string:

    extra_code = ""
    if special is not None:
        extra_code += r"""
            % yellow corner
            \draw[color=red!40!yellow, very thick] (C2) -- (C1) -- (C3);
            \node[draw, circle] at ($(C1) + (0.5cm,0.5cm)$) [color=red!40!yellow, very thick, inner sep=0.2ex] {{{}}};
        """.format(special)
    
    bm = boundary(-1, molecule, 4 - 1)
    bp = boundary(+1, molecule, 4 - 1)

    slcm = molecular_slices(4 - 1, bm)
    slcp = molecular_slices(4 - 1, bp)

    # slcm will be on top in the picture convention
    # which means it either has to be the 2 side or the 4 side
    # in the 23 and 14 moves. This is fine for the relations,
    # but does not faithfully represent things for the actual atoms
    if len(slcm) == 3 + 1 or len(slcm) == 1 + 1:
        slcm, slcp = slcp, slcm

    if len(slcm) == 2 + 1:
        m1 = tikz_code_for_2_molecule(slcm[0])
        m2 = tikz_code_for_2_molecule(slcm[1])
        m3 = tikz_code_for_2_molecule(slcm[2])
        m4 = tikz_code_for_2_molecule(slcp[1])
        m5 = tikz_code_for_2_molecule(slcp[2])

        if m1 == '' or m2 == '' or m3 == '' or m4 == '' or m5 == '':
            return ''

        code = r"""
        \begin{{tikzpicture}}[scale=0.5]

            \def\R{{4cm}}
            \def\clearance{{2mm}}
            \pgfmathsetlengthmacro{{\shorten}}
                {{(1cm+\clearance)/cos(36)}}
            \def\ang{{20}}
            \def\ext{{0.9cm}}

            \coordinate (P1) at (162:\R);
            \coordinate (P2) at (90:\R);
            \coordinate (P3) at (18:\R);
            \coordinate (P4) at (-126:\R);
            \coordinate (P5) at (-54:\R);
            \coordinate (L) at (132:{{\R + 2*\ext}});
            \coordinate (C1) at (-135:{{sqrt(2)*(\R+\ext)}});
            \coordinate (C2) at (-180+\ang : {{(\R+\ext)/cos(\ang)}});
            \coordinate (C3) at (-90-\ang : {{(\R+\ext)/cos(\ang)}});
            
            % number and label
            \node at (L) [] {{{}) {{\color{{blue}} 2-3}}}};{}

            % Pentagon arrows
            \draw[triple] ($(P1)!\shorten!(P2)$) -- ($(P2)!\shorten!(P1)$);
            \draw[triple] ($(P2)!\shorten!(P3)$) -- ($(P3)!\shorten!(P2)$);
            \draw[triple] ($(P1)!\shorten!(P4)$) -- ($(P4)!\shorten!(P1)$);
            \draw[triple] ($(P4)!\shorten!(P5)$) -- ($(P5)!\shorten!(P4)$);
            \draw[triple] ($(P5)!\shorten!(P3)$) -- ($(P3)!\shorten!(P5)$);
            \draw[quadruple,->] (0,0.45cm) -- (0,-0.65cm);

            \node at (P1) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P2) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P3) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P4) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P5) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};

        \end{{tikzpicture}}
        """.format(i, extra_code, m1, m2, m3, m4, m5)

    elif len(slcm) == 4 + 1:
        m1 = tikz_code_for_2_molecule(slcm[0])
        m2 = tikz_code_for_2_molecule(slcm[1])
        m3 = tikz_code_for_2_molecule(slcm[2])
        m4 = tikz_code_for_2_molecule(slcm[3])
        m5 = tikz_code_for_2_molecule(slcm[4])

        if m1 == '' or m2 == '' or m3 == '' or m4 == '' or m5 == '':
            return ''

        code = r"""
        \begin{{tikzpicture}}[scale=0.5]

            \def\R{{4cm}}
            \def\clearance{{2mm}}
            \pgfmathsetlengthmacro{{\shorten}}
                {{(1cm+\clearance)/cos(36)}}
            \def\ang{{20}}
            \def\ext{{0.9cm}}

            \coordinate (P1) at (-126:\R);
            \coordinate (P2) at (162:\R);
            \coordinate (P3) at (90:\R);
            \coordinate (P4) at (18:\R);
            \coordinate (P5) at (-54:\R);
            \coordinate (L) at (132:{{\R + 2*\ext}});
            \coordinate (C1) at (-135:{{sqrt(2)*(\R+\ext)}});
            \coordinate (C2) at (-180+\ang : {{(\R+\ext)/cos(\ang)}});
            \coordinate (C3) at (-90-\ang : {{(\R+\ext)/cos(\ang)}});

            % number and label
            \node at (L) [] {{{}) {{\color{{red}} 1-4}}}};{}

            % Pentagon arrows
            \draw[triple] ($(P1)!\shorten!(P2)$) -- ($(P2)!\shorten!(P1)$);
            \draw[triple] ($(P2)!\shorten!(P3)$) -- ($(P3)!\shorten!(P2)$);
            \draw[triple] ($(P3)!\shorten!(P4)$) -- ($(P4)!\shorten!(P3)$);
            \draw[triple] ($(P4)!\shorten!(P5)$) -- ($(P5)!\shorten!(P4)$);
            \draw[triple] ($(P1)!\shorten!(P5)$) -- ($(P5)!\shorten!(P1)$);
            \draw[quadruple,->] (0,0.45cm) -- (0,-0.65cm);

            \node at (P1) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P2) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P3) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P4) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};
            \node at (P5) {{\resizebox{{1cm}}{{1cm}}{{{}}}}};

        \end{{tikzpicture}}
        """. format(i, extra_code, m1, m2, m3, m4, m5)

    return code

#  ------------- Printing -------------

def represent_ogpos(signs: OGPos, mode: string = 'string', truncated: bool = False) -> dict[tuple | string, tuple[List[tuple | string], List[tuple | string]]]:
    """
    Convert a mask-based 'signs' dict (keys (e, s_mask), values ±1) into:
        simplex -> (neg_faces, pos_faces)

    - Keys (simplices) are encoded as tuples like (1,2,3) if mode='tuple',
      or strings like '123' if mode='string'.
    - Values are two lists of codimension-1 faces (encoded in the same 'mode'):
        neg_faces: faces s - {e} for which signs[(e, s)] = -1
        pos_faces: faces s - {e} for which signs[(e, s)] = +1

    Ordering:
      - simplices sorted by increasing size (|s|), then lex within size
      - faces sorted the same way

    Truncation:
      - If truncated=False (default), include all simplices of size >= 1 that appear
        as tops in 'signs' or as their faces.
      - If truncated=True, include only simplices with size >= 3 as keys (so seeded
        2-simplices and vertices are not shown as top-level keys). Face lists of
        included simplices may still contain size-2 faces.

    Returns:
      dict whose insertion order reflects the sorting above (Python 3.7+ preserves it).
    """
    if mode not in ('tuple', 'string'):
        raise ValueError("mode must be 'tuple' or 'string'")

    faces_dict = {s: ([],[]) for (_, s) in signs.keys() if not truncated or s.bit_count() >= 3}

    for (e, s), val in signs.items():
        if s not in faces_dict:
            continue
        face = s & ~(1 << e)
        if val == -1:
            faces_dict[s][0].append(face)
        elif val == +1:
            faces_dict[s][1].append(face)

    def simplex_sort(mask):
        return (mask.bit_count(), mask_to_tuple(mask))

    def encode(mask):
        if mode == 'tuple':
            s = mask_to_tuple(mask)
        elif mode == 'string':
            s = mask_to_string(mask)
        return s

    out = {}
    for s in sorted(faces_dict, key=simplex_sort):
        neg_sorted = sorted(faces_dict[s][0], key=simplex_sort)
        pos_sorted = sorted(faces_dict[s][1], key=simplex_sort)
        out[encode(s)] = ([encode(m) for m in neg_sorted],
                          [encode(m) for m in pos_sorted])

    return out


# get all solutions for a given N
#N = 4
#sols = list(atom_candidates(N))
#print(f"Number of n={} simplicial atoms: {}".format(N, len(sols)))


# get representatives for each Z_2^N orbit for a given N
N = 4
indep_sols = orbits_of_flip_action(N)
print("There are {} orbits of the flip action on 4-molecules.".format(len(indep_sols)))

code = ""
cnt = 0
cnt_spec = 0
for (i, sol) in enumerate(indep_sols):
    if cnt in [0,3,8,13,18,23,28,33,38,41,46,51,54,59,62,67,72,77]:
        code_add = tikz_code_for_4_molecule(cnt, sol, special=cnt_spec)
        cnt_spec += 1
    else:
        code_add = tikz_code_for_4_molecule(cnt, sol)
    
    cnt += 1
    
    code += code_add
    if i % 3 != 2:
        code += '%'

print(code)
