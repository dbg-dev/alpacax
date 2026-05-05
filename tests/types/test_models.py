import pytest
from pydantic import ValidationError

from alpacax.types.models import SplitConfig


def test_split_config_validates_weights():

    act = SplitConfig(train=0.5, val=0.4, test=0.1)
    assert act.train == 0.5
    assert act.val == 0.4
    assert act.test == 0.1

    with pytest.raises(ValidationError, match="splits must sum to 1.0"):
        SplitConfig(train=0.5, val=0.4, test=0.1001)

    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        SplitConfig(train=0.5, val=0.4, test=1.1001)

    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        SplitConfig(train=0.5, val=0.4, test=-0.0001)
