"""Tests tied to brief requirements the council flagged as untested: seed disjointness, chunk padding/masking at
episode end, runner re-planning cadence, replay-renders-before-step, goal never leaks the cube position."""
import json, os, numpy as np, torch, pytest
from pixelgrasp.env import EVAL_SEED_BASE, DEV_SEED_BASE, DEMO_SEED_BASE, goal_vec, policy_vec, privileged, make_env, PROPRIO_DIM, GOAL_DIM
from pixelgrasp.policy import BCPolicy, BCPolicyRunner
from pixelgrasp.train import DemoDataset


def test_seed_ranges_disjoint_and_ordered():
    assert EVAL_SEED_BASE < DEV_SEED_BASE < DEMO_SEED_BASE
    assert DEV_SEED_BASE - EVAL_SEED_BASE >= 1000 and DEMO_SEED_BASE - DEV_SEED_BASE >= 1000
    meta = "data/demos/meta.json"
    if os.path.exists(meta):
        assert json.load(open(meta))["seed_base"] >= DEMO_SEED_BASE


def test_policy_input_has_no_cube_position():
    env = make_env(); obs, _ = env.reset(seed=3); env.close()
    cube = privileged(obs)["obj"]
    for use_goal in (False, True):
        v = policy_vec(obs, use_goal)
        assert v.shape == (PROPRIO_DIM + (GOAL_DIM if use_goal else 0),)
        # no 3 consecutive entries equal the cube position
        assert not any(np.allclose(v[i:i + 3], cube, atol=1e-6) for i in range(len(v) - 2))
    assert np.allclose(policy_vec(obs, True)[-3:], goal_vec(obs))


def _tiny_dataset(tmp_path, chunk, use_goal=False):
    # two episodes of length 6 and 3; actions encode (episode, t) so padding is checkable
    rows = [(10000, t) for t in range(6)] + [(10001, t) for t in range(3)]
    n = len(rows)
    np.savez_compressed(tmp_path / "shard_0000.npz",
                        img=np.zeros((n, 96, 96, 3), np.uint8), proprio=np.zeros((n, 5), np.float32), goal=np.ones((n, 3), np.float32),
                        action=np.array([[e - 10000, t, 0, 1] for e, t in rows], np.float32),
                        episode=np.array([e for e, _ in rows], np.int32), t=np.array([t for _, t in rows], np.int16))
    return DemoDataset(str(tmp_path), chunk, use_goal=use_goal)


def test_chunk_padding_repeats_last_action_and_masks(tmp_path):
    ds = _tiny_dataset(tmp_path, chunk=4)
    _, _, act, mask = ds[4]            # episode 0, t=4: valid steps t=4,5 then padding
    assert mask.tolist() == [1, 1, 0, 0]
    assert act[:, 1].tolist() == [4, 5, 5, 5]      # padded steps repeat the final action
    assert (act[:, 0] == 0).all()                  # never crosses into episode 1
    _, _, act, mask = ds[6]            # episode 1, t=0: 3 valid
    assert mask.tolist() == [1, 1, 1, 0] and act[:, 0].tolist() == [1, 1, 1, 1]


def test_goal_concat_into_proprio(tmp_path):
    ds = _tiny_dataset(tmp_path, chunk=1, use_goal=True)
    _, pro, _, _ = ds[0]
    assert pro.shape == (PROPRIO_DIM + GOAL_DIM,) and pro[-3:].tolist() == [1, 1, 1]


def test_runner_replans_every_exec_steps(tmp_path):
    m = BCPolicy(chunk=8, pretrained=False)
    ck = tmp_path / "ck.pt"
    torch.save({"state_dict": m.state_dict(), "chunk": 8, "exec_steps": 4, "use_goal": False}, ck)
    r = BCPolicyRunner(str(ck), device="cpu")
    calls = []
    orig = r.model.act_chunk
    r.model.act_chunk = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    env = make_env(); obs, _ = env.reset(seed=0); img = env.render(); env.close()
    r.reset()
    for _ in range(9): r.act(obs, img)
    assert len(calls) == 3                        # steps 0, 4, 8
    with pytest.raises(AssertionError):
        BCPolicyRunner(str(ck), device="cpu", exec_steps=16)


def test_replay_renders_before_step():
    """check_data's replay must call render() before every step() to mirror collection (render perturbs warm-start)."""
    from pixelgrasp.check_data import replay_episode
    order = []
    class FakeEnv:
        def reset(self, seed): return {}, {}
        def render(self): order.append("r"); return None
        def step(self, a): order.append("s"); return None, 0, False, False, {"is_success": 1.0}
    assert replay_episode(FakeEnv(), 0, np.zeros((3, 4), np.float32)) is True
    assert order == ["r", "s", "r", "s", "r", "s"]
