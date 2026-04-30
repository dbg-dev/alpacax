from returns.result import Failure, Success

from alpacax.pipelines.instruction.stages import drain
from alpacax.types.errors import PipelineError, RowError


def test_drain_collects_successes():
    results = [
        Success("a"),
        Success("b"),
        Success("c"),
    ]

    oks, errs = drain(results)

    assert oks == ["a", "b", "c"]
    assert errs == []


def test_drain_collects_failures_as_pipeline_errors():
    results = [
        Failure(ValueError("bad")),
        Failure(RuntimeError("worse")),
    ]

    oks, errs = drain(results)

    assert oks == []
    assert len(errs) == 2
    assert isinstance(errs[0], PipelineError)
    assert isinstance(errs[1], PipelineError)
    assert str(errs[0]) == "bad"
    assert str(errs[1]) == "worse"


def test_drain_preserves_row_errors():
    row_error = RowError(row_index=3, error=ValueError("invalid row"))

    results = [
        Failure(row_error),
    ]

    oks, errs = drain(results)

    assert oks == []
    assert errs == [row_error]


def test_drain_preserves_pipeline_errors():
    pipeline_error = PipelineError(ValueError("network failed"))

    results = [
        Failure(pipeline_error),
    ]

    oks, errs = drain(results)

    assert oks == []
    assert errs == [pipeline_error]


def test_drain_collects_mixed_results_in_order():
    row_error = RowError(row_index=2, error=ValueError("missing field"))

    results = [
        Success("first"),
        Failure(ValueError("bad")),
        Success("second"),
        Failure(row_error),
    ]

    oks, errs = drain(results)

    assert oks == ["first", "second"]

    assert len(errs) == 2
    assert isinstance(errs[0], PipelineError)
    assert str(errs[0]) == "bad"

    assert errs[1] is row_error


def test_drain_accepts_generator():
    def stream():
        yield Success(1)
        yield Failure(ValueError("bad"))
        yield Success(2)

    oks, errs = drain(stream())

    assert oks == [1, 2]
    assert len(errs) == 1
    assert isinstance(errs[0], PipelineError)
    assert str(errs[0]) == "bad"
