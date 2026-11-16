import numpy as np, torch, pytest
from pixelgrasp.env import make_env, Expert, proprio, privileged, classify_failure, PROPRIO_DIM, ACTION_DIM, MAX_STEPS, IMG
from pixelgrasp.policy import BCPolicy, loss_fn
from pixelgrasp.eval import evaluate, ExpertPolicy, ZeroPolicy, wilson


def test_env_obs_layout():
    env = make_env(); obs, _ = env.reset(seed=0)
    assert proprio(obs).shape == (PROPRIO_DIM,)
    assert env.action_space.shape == (ACTION_DIM,)
    img = env.render(); assert img.shape == (IMG, IMG, 3) and img.dtype == np.uint8
    env.close()


def test_same_seed_same_scene():
    env = make_env(); a, _ = env.reset(seed=7); b, _ = env.reset(seed=7)
    assert np.allclose(privileged(a)["obj"], privileged(b)["obj"]) and np.allclose(a["desired_goal"], b["desired_goal"])
    c, _ = env.reset(seed=8); assert not np.allclose(privileged(a)["obj"], privileged(c)["obj"])
    env.close()


def test_expert_succeeds_on_eval_seeds():
    s, rows = evaluate(ExpertPolicy(), n_seeds=5, seed_base=0)
    assert s["success_rate"] == 1.0 and all(r["cause"] == "success" for r in rows)


def test_zero_policy_fails_and_is_classified():
    s, rows = evaluate(ZeroPolicy(), n_seeds=5, seed_base=100)
    assert s["success_rate"] <= 0.2 and set(s["causes"]) <= {"never_reached", "success"}


def test_policy_forward_shapes_and_ranges():
    m = BCPolicy(chunk=4, pretrained=False).eval()
    assert 11.0e6 < sum(p.numel() for p in m.parameters()) < 12.0e6     # README says 11.4 M
    img = torch.randint(0, 255, (2, IMG, IMG, 3), dtype=torch.uint8); p = torch.zeros(2, PROPRIO_DIM)
    pos, g = m(img, p)
    assert pos.shape == (2, 4, 3) and g.shape == (2, 4) and pos.abs().max() <= 1.0 and g.abs().max() < 100
    a = m.act_chunk(img[0].numpy(), p[0].numpy(), "cpu"); assert a.shape == (4, 4) and set(np.unique(a[:, 3])) <= {-1.0, 1.0}


def test_loss_masks_padded_steps():
    pos = torch.zeros(1, 2, 3); g = torch.zeros(1, 2); act = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1]]], dtype=torch.float32)
    full, _, _ = loss_fn(pos, g, act, torch.ones(1, 2)); half, _, _ = loss_fn(pos, g, act, torch.tensor([[1.0, 0.0]]))
    assert torch.isclose(full, half)   # masked step contributes nothing, mean over valid steps is identical


def test_failure_taxonomy():
    goal = np.array([1.4, 0.7, 0.6]); obj0 = np.array([1.3, 0.7, 0.425])
    far = np.tile(obj0, (10, 1)); grip_far = far + 0.2
    assert classify_failure({"is_success": 0}, far, grip_far, goal) == "never_reached"
    assert classify_failure({"is_success": 0}, far, far.copy(), goal) == "never_grasped"
    lifted = far.copy(); lifted[5, 2] += 0.1
    assert classify_failure({"is_success": 0}, lifted, lifted.copy(), goal) == "dropped"
    assert classify_failure({"is_success": 1}, far, far, goal) == "success"


def test_wilson():
    lo, hi = wilson(100, 100); assert lo > 0.95 and hi == 1.0
    lo, hi = wilson(50, 100); assert 0.40 < lo < 0.5 < hi < 0.60
