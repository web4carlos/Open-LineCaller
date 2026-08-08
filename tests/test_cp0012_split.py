from linecaller.dataset.splitting import split_clips


def test_split_is_deterministic():
    ids = [f"clip{i}" for i in range(20)]

    a = split_clips(ids, seed=42)
    b = split_clips(ids, seed=42)

    assert a == b
    assert len(a.train) + len(a.val) + len(a.test) == 20


def test_split_sets_are_disjoint():
    ids = [f"clip{i}" for i in range(20)]
    s = split_clips(ids, seed=1)

    assert not (set(s.train) & set(s.val))
    assert not (set(s.train) & set(s.test))
    assert not (set(s.val) & set(s.test))
