"""欧氏锥的整数圆盘截面，局部邻居与摘要求和共用同一边界。"""

from functools import lru_cache
from math import isqrt


@lru_cache(maxsize=None)
def euclidean_rows(radius: int, horizon: int) -> tuple[tuple[int, int, int], ...]:
    """返回 (过去距离, 行偏移, 行半宽)，不对连续圆盘半径提前取整。"""
    if type(radius) is not int or type(horizon) is not int or radius < 0 or horizon < 1:
        raise ValueError("Euclidean cone requires nonnegative integer radius and positive horizon")
    rows = []
    for distance in range(1, horizon + 1):
        bound = radius * distance // horizon
        for y in range(-bound, bound + 1):
            remaining = (radius * distance) ** 2 - (horizon * y) ** 2
            half_width = isqrt(remaining // horizon**2)
            rows.append((distance, y, half_width))
    return tuple(rows)


def euclidean_offsets(radius: int, horizon: int) -> tuple[tuple[int, int, int], ...]:
    """过去方向固定为负时间；每个圆盘格点仅出现一次。"""
    return tuple((x, y, -distance)
                 for distance, y, half_width in euclidean_rows(radius, horizon)
                 for x in range(-half_width, half_width + 1))
