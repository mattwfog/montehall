"""Integration check against prepared research artifacts; unit tests need no downloads."""

from pathlib import Path

from soccerviz.ui.workbench import build_app


def main():
    app = build_app(Path("data"), Path("artifacts"))
    checks = {
        "video_radar": [99],
        "tactical_snapshot": ["g2-f58880"],
        "positioning_scenario": ["g2-f58880", "Home:Player11", 3, 0],
        "probabilistic_forecast": [62007],
        "ball_forecast": [3073],
        "missing_players": [100],
        "replay": [2, "g2-p0000", 6, True],
    }
    completed = []
    for function in app.fns.values():
        if function.api_name in checks:
            result = function.fn(*checks[function.api_name])
            if result is None:
                raise AssertionError(f"No result from {function.api_name}")
            completed.append(function.api_name)
            print(f"{function.api_name}: OK")
    if set(completed) != set(checks):
        raise AssertionError(f"Missing research endpoints: {set(checks) - set(completed)}")
    print(f"Verified {len(completed)} workbench data flows")


if __name__ == "__main__":
    main()
