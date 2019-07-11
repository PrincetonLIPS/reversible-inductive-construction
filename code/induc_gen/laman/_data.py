import typing


class LamanSamplerConfig(typing.NamedTuple):
    expected_corruption_steps: int
    use_revisit: bool
    num_steps: int
    max_denoising_steps: int = 20
