"""Discover the six history grids from a teacher60 export's actual graph.

Tensor numbers are export-local. In particular, READ_VARIABLE order differs
between the published Willow and Nyx exports. This helper never invokes a model.
"""


def state_grid_layout(operations, grid):
    reads = [op for op in operations if op["op"] == "READ_VARIABLE"]
    writes = [op for op in operations if op["op"] == "ASSIGN_VARIABLE"]
    assert len(reads) == len(writes) == 6, "Expected six history states"
    read_handles = {op["outputs"][0]: op["inputs"][0] for op in reads}
    assignments = {op["inputs"][0]: op["inputs"][1] for op in writes}
    assert len(read_handles) == len(set(read_handles.values())) == len(assignments) == 6
    assert set(read_handles.values()) == assignments.keys()
    producers = {}
    for op in operations:
        for index in op["outputs"]:
            assert index not in producers
            producers[index] = op

    def same_grid(*indices):
        values = [grid(index) for index in indices]
        for value in values:
            assert value["scales"] and all(s > 0 for s in value["scales"])
        for value in values[1:]:
            for key in ("dtype", "scales", "zeros"):
                assert value[key] == values[0][key], (indices, key)

    bindings = []
    for concat in (op for op in operations if op["op"] == "CONCATENATION"):
        read, = [i for i in concat["inputs"] if i in read_handles]
        current, = [i for i in concat["inputs"] if i != read]
        assert concat["inputs"] == [read, current], "Expected history before current features"
        output, = concat["outputs"]
        handle = read_handles[read]
        assigned = assignments[handle]
        writer = producers[assigned]
        assert writer["op"] == "STRIDED_SLICE" and writer["inputs"][0] == output
        source = current
        if producers[current]["op"] == "EXPAND_DIMS":
            source = producers[current]["inputs"][0]
            same_grid(source, current)
        same_grid(source, read, output, assigned)
        old, new, joined, kept = [grid(i)["shape"] for i in (read, current, output, assigned)]
        assert len(old) == len(new) == len(joined) == len(kept) == 4
        assert old == kept and old[0] == new[0] and old[2:] == new[2:]
        assert joined == [old[0], old[1] + new[1], *old[2:]]
        bindings.append(dict(handle=handle, producer=source, current=current,
                             read=read, concatenation=output, assigned=assigned))
    assert len(bindings) == len({b["handle"] for b in bindings}) == 6
    concat_outputs = {b["concatenation"] for b in bindings}
    reshapes = []
    for op in operations:
        if op["op"] == "RESHAPE" and op["inputs"][0] in concat_outputs:
            source, output = op["inputs"][0], op["outputs"][0]
            same_grid(source, output)
            reshapes.append([source, output])
    assert len(reshapes) == 1, "Expected one final context flatten"
    return dict(state_grid_pairs=[[b["producer"], b["read"], b["concatenation"]]
                                 for b in bindings],
                state_bindings=bindings, reshape_grid_pairs=reshapes)
