from montehall_cv.store.vlm_cache import VlmCache, content_key


def test_content_key_stable_and_distinct():
    a = content_key("claude-sonnet-5", "PROMPT", b"\x01\x02\x03")
    assert a == content_key("claude-sonnet-5", "PROMPT", b"\x01\x02\x03")
    assert a != content_key("claude-sonnet-5", "PROMPT", b"\x01\x02\x04")
    # delimiter prevents concatenation collisions: ("ab","c") != ("a","bc")
    assert content_key("ab", "c") != content_key("a", "bc")


def test_put_get_and_persist_across_reopen(tmp_path):
    path = tmp_path / "cache" / "rims.jsonl"
    c = VlmCache(path)
    assert c.get("k1") is None
    c.put("k1", {"candidates": [{"x1": 0.1}]})
    assert c.get("k1") == {"candidates": [{"x1": 0.1}]}

    # a fresh instance (re-run after crash) recovers the persisted verdict
    reopened = VlmCache(path)
    assert reopened.get("k1") == {"candidates": [{"x1": 0.1}]}


def test_torn_final_line_tolerated(tmp_path):
    path = tmp_path / "cache.jsonl"
    c = VlmCache(path)
    c.put("good", {"v": 1})
    with path.open("a") as fh:
        fh.write('{"key": "partial", "value": {"v": 2')  # no newline, truncated
    reopened = VlmCache(path)
    assert reopened.get("good") == {"v": 1}
    assert reopened.get("partial") is None
