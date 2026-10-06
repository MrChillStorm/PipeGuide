import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pipefit  # noqa: E402
import synthetic  # noqa: E402


def fit(kind, noise=0.0, **kw):
    pts, tris = synthetic.BODIES[kind](noise=noise)
    prof = pipefit.sample_profile(pts, tris)
    chain = pipefit.fit_chain(prof.x, prof.channels("area"), **kw)
    return prof, chain


def test_profile_area_matches_geometry():
    pts, tris = synthetic.liner(n_theta=96)
    prof = pipefit.sample_profile(pts, tris)
    mid = len(prof.x) // 2
    assert prof.area[mid] == pytest.approx(np.pi, rel=0.01)
    assert np.allclose(prof.centroid[mid], 0, atol=1e-6)


def test_simple_body_needs_few_pipes():
    _, chain = fit("liner")
    assert chain.n_segments <= 8


def test_beats_legacy_pipe_budget_without_losing_fit():
    prof, chain = fit("glider")
    ev = pipefit.evaluate_sections(prof, pipefit.chain_to_sections(chain))
    assert chain.n_segments < 20
    # The round-section limit for this body is ~91.8% overlap.
    assert ev["iou"] > 0.91


def test_tighter_tolerance_never_uses_fewer_pipes():
    counts = [fit("glider", tol=t)[1].n_segments for t in (0.05, 0.01, 0.003)]
    assert counts == sorted(counts)


def test_noise_does_not_inflate_pipe_count():
    clean = fit("glider")[1].n_segments
    noisy = fit("glider", noise=0.004)[1].n_segments
    assert noisy <= clean + 3


def test_unreachable_tolerance_does_not_just_use_the_cap():
    _, chain = fit("glider", tol=1e-6, kmax=63)
    assert chain.n_segments < 40


def test_zero_tolerance_uses_requested_count():
    _, chain = fit("glider", tol=0, kmax=17)
    assert chain.n_segments == 17


def test_default_is_volume_neutral():
    prof, chain = fit("glider", noise=0.002)
    _, _, r = chain.at(prof.x)
    assert (np.pi * r ** 2).sum() / prof.area.sum() == pytest.approx(1.0, abs=0.01)


def test_bias_moves_volume_in_requested_direction():
    vols = []
    for b in (-1.0, 0.0, 1.0):
        prof, chain = fit("glider", bias=b)
        _, _, r = chain.at(prof.x)
        vols.append((np.pi * r ** 2).sum())
    assert vols[0] < vols[1] < vols[2]


def test_sections_are_contiguous_and_follow_yasim_convention():
    prof, chain = fit("glider")
    chain = chain.extended(*prof.x_range)
    secs = pipefit.chain_to_sections(chain)
    for a, b in zip(secs, secs[1:]):
        assert a[3:6] == pytest.approx(b[0:3])
    assert secs[0][0] == pytest.approx(-prof.x_range[0])
    assert secs[-1][3] == pytest.approx(-prof.x_range[1])
    for s in secs:
        assert s[6] > 0 and 0 < s[7] <= 1 and s[8] in (0.0, 0.5, 1.0)


def test_enclosing_fit_contains_more_than_area_fit():
    pts, tris = synthetic.flat_belly()
    prof = pipefit.sample_profile(pts, tris)
    area_r = prof.channels("area")[:, 2]
    enc_r = prof.channels("enclosing")[:, 2]
    assert (enc_r >= area_r - 1e-6)[prof.valid].all()


def test_evaluator_scores_perfect_pipe_perfectly():
    pts, tris = synthetic.liner(n_theta=128)
    prof = pipefit.sample_profile(pts, tris)
    chain = pipefit.fit_chain(prof.x, prof.channels("area"), tol=0, kmax=60)
    ev = pipefit.evaluate_sections(prof, pipefit.chain_to_sections(chain))
    assert ev["iou"] > 0.99


def test_empty_mesh_is_a_clean_error():
    with pytest.raises(ValueError):
        pipefit.sample_profile(np.zeros((3, 3)), np.array([[0, 1, 2]]))


def lobe_eval(kind, n, noise=0.002):
    pts, tris = synthetic.BODIES[kind](noise=noise)
    prof = pipefit.sample_profile(pts, tris)
    chain = pipefit.fit_lobes(prof, n).extended(*prof.x_range)
    secs = pipefit.chain_to_sections(chain)
    return pipefit.evaluate_sections(prof, secs), chain, secs


def test_lobes_capture_non_round_sections():
    ev, _, _ = lobe_eval("flat_belly", 3)
    assert ev["iou"] > 0.85          # a single round pipe is stuck at ~51%
    assert ev["under"] < 0.1 and ev["over"] < 0.1


def test_more_lobes_fit_better():
    ious = [lobe_eval("flat_belly", n)[0]["iou"] for n in (1, 3, 4)]
    assert ious == sorted(ious)


def test_lobes_are_pruned_on_round_bodies():
    _, _, secs = lobe_eval("liner", 3)
    assert len(secs) <= 8            # redundant lobes must not multiply pipes


def test_lobe_pipe_count_stays_bounded():
    _, chain, secs = lobe_eval("flat_belly", 3)
    assert len(secs) <= 3 * chain.n_segments


def balanced(kind, **kw):
    pts, tris = synthetic.BODIES[kind](noise=0.002)
    prof = pipefit.sample_profile(pts, tris)
    ch = pipefit.fit_balanced(prof, **kw)
    secs = pipefit.chain_to_sections(ch)
    return ch, secs, pipefit.evaluate_sections(prof, secs)


def test_balanced_keeps_round_bodies_minimal():
    ch, secs, ev = balanced("liner")
    assert ch.n_lobes == 1 and len(secs) <= 8 and ev["iou"] > 0.99


def test_balanced_adds_lobes_when_they_pay_for_themselves():
    ch, secs, ev = balanced("flat_belly")
    assert ch.n_lobes >= 3 and ev["iou"] > 0.85


def test_higher_pipe_price_means_fewer_pipes():
    cheap = len(balanced("flat_belly", pipe_cost=0.0005, fdm_weight=0)[1])
    dear = len(balanced("flat_belly", pipe_cost=0.02, fdm_weight=0)[1])
    assert dear < cheap


def test_fdm_weight_trades_shape_for_less_distortion():
    free = balanced("glider", fdm_weight=0.0)[0].info
    guarded = balanced("glider", fdm_weight=0.3)[0].info
    assert guarded["fdm_damage"] < free["fdm_damage"]
    assert guarded["shape_mismatch"] > free["shape_mismatch"]


def test_single_pipe_layout_has_zero_fdm_damage_against_itself():
    pts, tris = synthetic.glider()
    prof = pipefit.sample_profile(pts, tris)
    secs = pipefit.chain_to_sections(
        pipefit.fit_chain(prof.x, prof.channels("area")).extended(*prof.x_range))
    damage, det = pipefit.fdm_damage(secs, secs, prof.x_range)
    assert damage == pytest.approx(0.0, abs=1e-9)
    assert det["drag_ratio"] == pytest.approx(1.0)


def test_fdm_damage_ignores_uniform_drag_scaling_but_not_redistribution():
    base = [(0, 0, 0, 4, 0, 0, 1.0, 1.0, 0.5)]
    # Same pipe twice over the whole length: drag doubles uniformly.
    twice = base * 2
    _, uniform = pipefit.fdm_damage(twice, base, (-4, 0))
    assert uniform["drag_shift"] == pytest.approx(0.0, abs=1e-9)
    assert uniform["drag_ratio"] == pytest.approx(2.0)
    # Extra pipe over just the first half: drag is redistributed.
    half = base + [(0, 0, 0, 2, 0, 0, 1.0, 1.0, 0.5)]
    moved, det = pipefit.fdm_damage(half, base, (-4, 0))
    assert det["drag_shift"] > 0.1


def test_lobe_row_runs_across_the_wide_direction():
    pts, tris = synthetic.flat_belly()
    assert pipefit.lobe_axis(pipefit.sample_profile(pts, tris)) == "y"


def test_yasim_totals_match_hand_computation():
    # One unit-width, length-2 cylinder: 2 segments, each weight 1.
    t = pipefit.yasim_totals([(0, 0, 0, 2, 0, 0, 1.0, 1.0, 0.5)])
    assert t["surfaces"] == 2 and t["contacts"] == 2
    assert t["drag"] == pytest.approx(2.0) and t["mass"] == pytest.approx(2.0)


def test_lobes_inflate_yasim_drag_only_where_pipes_overlap():
    def drag(kind, n):
        pts, tris = synthetic.BODIES[kind](noise=0.002)
        prof = pipefit.sample_profile(pts, tris)
        ch = (pipefit.fit_lobes(prof, n) if n > 1
              else pipefit.fit_chain(prof.x, prof.channels("area")))
        return pipefit.yasim_totals(pipefit.chain_to_sections(ch.extended(*prof.x_range)))["drag"]
    assert drag("liner", 3) == pytest.approx(drag("liner", 1), rel=0.05)
    assert drag("flat_belly", 3) > 1.5 * drag("flat_belly", 1)


@pytest.mark.parametrize("stations", [400, 504, 640])
def test_pipe_count_is_stable_across_station_counts(stations):
    # Regression: a borderline error percentile once flipped this between
    # ~20 and 180 pipes depending only on the sampling density.
    pts, tris = synthetic.flat_belly()
    prof = pipefit.sample_profile(pts, tris, n_stations=stations)
    chain = pipefit.fit_lobes(prof, 3)
    assert len(pipefit.chain_to_sections(chain)) <= 30


def aero(kind, n):
    pts, tris = synthetic.BODIES[kind](noise=0.002)
    prof = pipefit.sample_profile(pts, tris)
    ch = (pipefit.fit_lobes(prof, n) if n > 1
          else pipefit.fit_chain(prof.x, prof.channels("area")))
    return pipefit.aero_error(prof, pipefit.chain_to_sections(ch.extended(*prof.x_range)))


def test_aero_geometry_of_a_cylinder_is_exact():
    # Cylinder r=1, length 4 along x.
    pts, tris = synthetic.liner(n_theta=128)
    prof = pipefit.sample_profile(pts, tris)
    g = pipefit.aero_geometry(prof)
    assert g["max_section"] == pytest.approx(np.pi, rel=0.01)
    assert g["wetted"] > 0 and g["side_area"] > 0 and g["plan_area"] > 0


def test_round_body_one_pipe_matches_aero_geometry():
    assert aero("liner", 1)["rms"] < 0.02


def test_single_pipe_misstates_airflow_geometry_of_flat_body():
    one, lobes = aero("flat_belly", 1), aero("flat_belly", 4)
    assert abs(one["side_area"]) > 0.5 and one["plan_area"] < -0.25
    assert lobes["rms"] < 0.05 < one["rms"]


def test_balanced_default_reports_aero_error():
    ch, secs, ev = balanced("flat_belly")
    assert ch.info["aero"]["rms"] < 0.05
