from autopass.physics import validate_logical_world
from autopass.scenarios import all_comprehensive_scenarios
from visual_world import initialize_world


def test_all_comprehensive_scenarios_have_valid_initial_layout():
    issues_total = 0
    for spec in all_comprehensive_scenarios():
        world = initialize_world(spec)
        issues = validate_logical_world(spec, world)
        assert not issues, f"{spec.scenario_id}: {issues}"
        issues_total += len(issues)
    assert len(all_comprehensive_scenarios()) == 18


def test_lead_starts_ahead_of_ego():
    for spec in all_comprehensive_scenarios():
        world = initialize_world(spec)
        assert world.lead_x_m > world.ego_x_m + 2.0, spec.scenario_id


def test_rear_starts_behind_ego():
    for spec in all_comprehensive_scenarios():
        world = initialize_world(spec)
        assert world.rear_x_m < world.ego_x_m - 3.0, spec.scenario_id


def test_oncoming_starts_ahead_in_passing_lane():
    for spec in all_comprehensive_scenarios():
        world = initialize_world(spec)
        assert world.oncoming_x_m > world.ego_x_m + 3.0, spec.scenario_id


def test_kinematic_steps_stay_physically_valid():
    from visual_world import advance_world_step

    for spec in all_comprehensive_scenarios()[:6]:
        world = initialize_world(spec)
        for action in ("wait", "wait", "pass", "wait"):
            world = advance_world_step(spec, world, action=action, dt=1.0)
            issues = validate_logical_world(spec, world)
            if world.collision:
                assert world.done
                break
            assert not issues, f"{spec.scenario_id} after {action}: {issues}"
            if world.done:
                break
