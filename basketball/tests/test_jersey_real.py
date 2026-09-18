from montehall_cv.training.build_jersey_real import (
    bind_roster_cluster,
    is_truncation_ambiguous,
    normalize_number,
    roster_entry,
    roster_number_set,
    top_posteriors,
    verified_tracks,
)


def test_roster_entry_accepts_map_and_bare_forms():
    bare = {"team_id": 1, "team_name": "X", "players": [{"number": "5", "name": None}]}
    assert roster_entry(bare, "any-job") is bare
    keyed = {"job-a": bare}
    assert roster_entry(keyed, "job-a") is bare
    assert roster_entry(keyed, "job-b") is None


def test_normalize_matches_pipeline_canonical_form():
    assert normalize_number("05") == "5"
    assert normalize_number(" 23 ") == "23"
    assert normalize_number("100") is None
    assert normalize_number("A5") is None


def test_roster_set_drops_unusable_numbers():
    players = [{"number": "05"}, {"number": "23"}, {"number": "TBD"}]
    assert roster_number_set(players) == {"5", "23"}


def test_truncation_ambiguity_is_containment_not_prefix():
    roster = {"1", "5", "10", "23", "24", "77"}
    # "52"->"5" style collisions: digit appears inside a 2-digit number
    assert is_truncation_ambiguous("1", roster)  # 10
    assert is_truncation_ambiguous("2", roster)  # 23, 24
    assert is_truncation_ambiguous("3", roster)  # 23 (second digit)
    assert is_truncation_ambiguous("7", roster)  # 77
    assert not is_truncation_ambiguous("5", roster)
    assert not is_truncation_ambiguous("9", roster)
    assert not is_truncation_ambiguous("23", roster)  # 2-digit reads never ambiguous


def test_top_posteriors_takes_best_candidate_above_bar():
    rows = [
        {"track_id": 1, "candidate": "23", "prob": 0.8},
        {"track_id": 1, "candidate": "28", "prob": 0.2},
        {"track_id": 2, "candidate": "5", "prob": 0.4},
    ]
    assert top_posteriors(rows, 0.6) == {1: "23"}


def test_bind_requires_margin_and_min_matches():
    roster = {"5", "9", "23"}
    cluster_of = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    # cluster 0 has two in-roster reads, cluster 1 none -> binds 0
    reads = {1: "5", 2: "23", 3: "88", 4: "77", 5: "9"}
    bound, counts = bind_roster_cluster(reads, cluster_of, roster)
    assert bound == 0
    assert counts == {0: 2, 1: 0}
    # cluster 5 (=2, refs) never participates even with an in-roster read
    # tie -> no binding
    reads_tied = {1: "5", 3: "23"}
    bound, _ = bind_roster_cluster(reads_tied, cluster_of, roster)
    assert bound is None
    # thin evidence -> no binding
    bound, _ = bind_roster_cluster({1: "5"}, cluster_of, roster)
    assert bound is None


def test_attribution_filter_is_removal_only_and_bound_cluster_scoped():
    from montehall_cv.roster import attribution_allowed

    roster = {"5", "23", "24"}
    # bound cluster: impossible numbers and ambiguous single digits blocked
    assert not attribution_allowed("88", 0, 0, roster)   # not on roster
    assert not attribution_allowed("2", 0, 0, roster)    # ambiguous (23, 24)
    assert attribution_allowed("23", 0, 0, roster)
    assert attribution_allowed("5", 0, 0, roster)
    # opponent cluster has no roster — everything passes
    assert attribution_allowed("88", 1, 0, roster)
    # no binding -> filter disabled entirely
    assert attribution_allowed("88", 0, None, roster)


def test_verified_tracks_filters_cluster_roster_and_ambiguity():
    roster = {"5", "23", "24"}
    cluster_of = {1: 0, 2: 0, 3: 0, 4: 1}
    reads = {1: "5", 2: "2", 3: "88", 4: "23"}
    # track 2: "2" ambiguous (23, 24); track 3: not in roster; track 4: wrong cluster
    assert verified_tracks(reads, cluster_of, roster, bound_cluster=0) == {1: "5"}
