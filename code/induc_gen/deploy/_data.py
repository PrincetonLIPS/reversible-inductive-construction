import typing


class SampleResult(typing.NamedTuple):
    """ This class represents the result of running a chain.

    seed: the seed of the chain
    expected_corruption_step: the number of expected corruptions to be applied at each step.
    x: a list of sampled molecules.
    x_tilde: a list of sampled corruptions.
    corruption_steps: the actual number of taken corruption steps.
    denoising_steps: the actual number of taken denoising steps.
    stop_by_revisit: whether the chain was stopped by revisit at each step.
    meta: metadata for the sampled chain.
    """
    seed: int
    expected_corruption_step: int
    x: typing.List[str]
    x_tilde: typing.List[str]
    corruption_steps: typing.List[int]
    denoising_steps: typing.List[int]
    transition_attempts: typing.List[int]
    stop_by_revisit: typing.List[bool]
    meta: typing.Dict[str, typing.Any]
