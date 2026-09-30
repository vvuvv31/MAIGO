"""Exact shared-prefix graph of homogeneous-water electron-family paths.

Nodes are points in the ROOT electron's parent frame, with a predecessor.
Sharing prefixes avoids storing the same trajectory once per depositing step.
No approximation, pruning, or endpoint-only escape assumption is made here.
"""
import numpy as np

NONE = np.uint32(0xffffffff)
NODE_DTYPE = np.dtype([('previous', '<u4'), ('reserved', '<u4'), ('point', '<f8', (3,))])


def parent_basis(direction):
    z = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(z)
    if not np.isfinite(norm) or abs(norm-1) > 1e-6:
        raise ValueError('Invalid parent direction')
    z = z/norm
    helper = np.eye(3)[np.argmin(np.abs(z))]
    x = np.cross(helper, z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def build_path_graph(d, first, row_roots, *, source_rows=None, state_links=None):
    # Optional V4 companion: node state and incoming edge may belong to
    # different tracks at a birth node. Do not assign the child's momentum
    # to its parent's incoming segment.
    if (source_rows is None) != (state_links is None):
        raise ValueError('Source rows and state-link output must be supplied together')
    if source_rows is not None:
        source_rows = np.asarray(source_rows)
        if (d.shape[1] != 46 or source_rows.shape != (len(d),)
                or source_rows.dtype.kind not in 'ui' or np.any(source_rows < 0)
                or len(np.unique(source_rows)) != len(d) or state_links):
            raise ValueError('Invalid V4 source-row mapping')
    starts = np.r_[0, np.flatnonzero(np.any(d[1:, :3] != d[:-1, :3], axis=1))+1]
    ends = np.r_[starts[1:], len(d)]
    if len(starts) != len(first) or not np.array_equal(d[starts, :3], first[:, :3]):
        raise ValueError('Track ordering differs from family audit')
    keys = [tuple(r[:3].astype(int)) for r in first]
    lookup = {key: i for i, key in enumerate(keys)}
    capacity = int(np.count_nonzero(row_roots >= 0) + len(first))
    if capacity >= int(NONE):
        raise ValueError('Path graph exceeds uint32 domain')
    graph = np.zeros(capacity, dtype=NODE_DTYPE)
    if source_rows is not None:
        node_rows = np.full(capacity, np.iinfo(np.uint64).max, dtype='<u8')
        edge_rows = node_rows.copy()
        node_post = np.zeros(capacity, dtype=np.uint8)
    pre = np.full(len(d), NONE, dtype=np.uint32)
    post = np.full(len(d), NONE, dtype=np.uint32)
    state = np.zeros(len(first), dtype=np.uint8)
    size = 0
    max_birth_error = 0.

    def add(point, previous):
        nonlocal size
        if size >= capacity:
            raise ValueError('Path graph capacity accounting')
        graph[size] = (previous, 0, point)
        size += 1
        return size-1

    def build(i):
        nonlocal size, max_birth_error
        if state[i] == 2:
            return
        if state[i] == 1:
            raise ValueError('Cyclic ancestry')
        state[i] = 1
        begin, end = starts[i], ends[i]
        rows = d[begin:end]
        root_index = row_roots[begin]
        if root_index < 0:
            state[i] = 2
            return
        if (np.any(np.diff(rows[:, 5]) != 1)
                or (len(rows)>1 and np.max(np.linalg.norm(rows[1:, 10:13]-rows[:-1, 13:16], axis=1)) > 1e-6)
                or np.linalg.norm(rows[0, 6:9]-rows[0, 10:13]) > 1e-6):
            raise ValueError('Discontinuous track')
        root = first[root_index]
        basis = parent_basis(root[24:27])
        origin = root[6:9]
        pk = (*keys[i][:2], int(first[i, 3]))
        if pk not in lookup:
            raise ValueError('Missing parent')
        parent = lookup[pk]
        if first[parent, 3] == 0:
            if i != root_index or first[i, 4] != 11:
                raise ValueError('Not an electron root')
            head = add(np.zeros(3), NONE)
        else:
            build(parent)
            p = d[starts[parent]:ends[parent]]
            delta = p[:, 13:16]-p[:, 10:13]
            length2 = np.sum(delta**2, axis=1)
            t = np.divide(np.sum((rows[0, 6:9]-p[:, 10:13])*delta, axis=1),
                          length2, out=np.zeros(len(p)), where=length2>0)
            t = np.clip(t, 0, 1)
            error = np.linalg.norm(p[:, 10:13]+t[:, None]*delta-rows[0, 6:9], axis=1)
            if d.shape[1] == 46:
                # V4 complete genealogy is authoritative. A geometric nearest
                # match can pick a different visit to the same position.
                if first[i, 22] != 1 or first[i, 29] <= 0:
                    raise ValueError('Missing exact V4 generating step')
                matched = np.flatnonzero(p[:, 5] == first[i, 29])
                if len(matched) != 1:
                    raise ValueError('Missing/ambiguous V4 generating step')
                j = int(matched[0])
            else:
                j = int(np.argmin(error))
            if error[j] > 1e-6:
                raise ValueError('Birth not on parent polyline')
            hits = np.flatnonzero(error < 1e-8)
            if d.shape[1] != 46 and len(hits)>1 and np.linalg.norm(delta[hits.min()+1:hits.max()], axis=1).sum()>1e-6:
                raise ValueError('Ambiguous parent birth position')
            max_birth_error = max(max_birth_error, float(error[j]))
            # The child's birth can lie inside a parent step. Its predecessor
            # must be BEFORE that step, never the parent's future endpoint.
            head = add(basis @ (rows[0, 6:9]-origin), pre[starts[parent]+j])
        n = end-begin
        if source_rows is not None:
            node_rows[head] = source_rows[begin]  # birth uses child PRE state
            if first[parent, 3] != 0:
                edge_rows[head] = source_rows[starts[parent]+j]
            node_rows[size:size+n] = source_rows[begin:end]
            edge_rows[size:size+n] = source_rows[begin:end]
            node_post[size:size+n] = 1
        ids = np.arange(size, size+n, dtype=np.uint32)
        pre[begin:end] = np.r_[np.uint32(head), ids[:-1]]
        post[begin:end] = ids
        graph['previous'][size:size+n] = pre[begin:end]
        graph['point'][size:size+n] = (rows[:, 13:16]-origin) @ basis.T
        size += n
        state[i] = 2

    for i in range(len(first)):
        build(i)
    graph = graph[:size].copy()
    if np.any((graph['previous'] != NONE) & (graph['previous'] >= np.arange(size))):
        raise ValueError('Non-topological path graph')
    if source_rows is not None:
        state_links.update(node_source_row=node_rows[:size].copy(),
                           incoming_edge_source_row=edge_rows[:size].copy(),
                           node_is_post_state=node_post[:size].copy())
    return graph, pre, dict(nodes=int(size), maximum_birth_error_mm=max_birth_error)


def backward_points(graph, head):
    """Reference traversal for tests; production uses the identical links."""
    points = []
    while head != NONE:
        if head < 0 or head >= len(graph):
            raise ValueError('Out-of-range path link')
        points.append(graph['point'][head].copy())
        previous = int(graph['previous'][head])
        if previous != int(NONE) and previous >= head:
            raise ValueError('Non-topological path link')
        head = previous
    return np.asarray(points).reshape(-1, 3)
