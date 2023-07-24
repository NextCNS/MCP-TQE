from dwave.system import DWaveSampler, EmbeddingComposite

def test():
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

    linear = {('a', 'a'): -1, ('b', 'b'): -1, ('c', 'c'): -1}
    quadratic = {('a', 'b'): 2, ('b', 'c'): 2, ('a', 'c'): 2}
    Q = {**linear, **quadratic}
