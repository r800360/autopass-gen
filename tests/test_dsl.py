from autopass.dsl import PassingDSL, PerceptionRecord, VerificationNote, init_dsl_from_request


def test_dsl_starts_with_empty_tool_queue():
    dsl = init_dsl_from_request("hurry to airport", aggression="high")
    assert dsl.tools_pending == []
    assert dsl.tools_completed == []


def test_dsl_appends_perception_and_marks_tool_done():
    dsl = init_dsl_from_request("hurry to airport", aggression="high")
    rec = PerceptionRecord(tool="capture_sensors", summary="ok", data={"x": 1})
    dsl2 = dsl.append_perception(rec)
    assert "capture_sensors" in dsl2.tools_completed


def test_replan_clears_tools_without_default_queue():
    dsl = init_dsl_from_request("test")
    dsl = dsl.append_perception(PerceptionRecord(tool="capture_sensors", summary="ok", data={}))
    note = VerificationNote(verdict="replan", message="traffic", revision_triggered=True)
    dsl2 = dsl.append_verification(note)
    assert dsl2.revision == 1
    assert dsl2.tools_completed == []
    assert dsl2.tools_pending == []
