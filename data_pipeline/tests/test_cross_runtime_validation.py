import pytest
from data_pipeline.validate_r2r import assert_label_equivalent


def test_cross_blas_roundoff_does_not_reject_identical_pointing():
    assert_label_equivalent({"pixel": [163.24695910109799, 217.50374249885928], "token_id": 821, "loss_valid": True},
                           {"pixel": [163.24695910109799, 217.5037424988593], "token_id": 821, "loss_valid": True})


@pytest.mark.parametrize("change", [{"token_id": 820}, {"loss_valid": False}, {"pixel": [163.24696, 217.503]}])
def test_real_label_changes_are_rejected(change):
    label = {"pixel": [163.24695910109799, 217.50374249885928], "token_id": 821, "loss_valid": True}
    with pytest.raises(AssertionError):
        assert_label_equivalent({**label, **change}, label)
