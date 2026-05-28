from autopass.executor import execute_step, should_use_carla_vehicle
from autopass.dsl import init_dsl_from_request
from visual_world import curated_demo_scenarios, initialize_world


def test_vehicle_control_not_used_in_test_mode():
    assert should_use_carla_vehicle("visual") is False


def test_kinematic_execute_appends_execution_and_belief():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text)
    after, dsl2, fb = execute_step(spec, world, dsl, "wait", backend="visual")
    assert fb["mode"] == "kinematic"
    assert len(dsl2.execution_log) == 1
    assert dsl2.world_belief.source == "visual_depth"
    assert after.t_s > world.t_s
