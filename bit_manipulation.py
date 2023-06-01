from functools import reduce
from typing import List


class Solution:
    # 461
    def hammingDistance(self, x: int, y: int) -> int:
        return (x ^ y).bit_count()

    # 136
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda last, num: last ^ num, nums, 0)

    # 268
    def missingNumber(self, nums: List[int]) -> int:
        return reduce(lambda last, i: last ^ i ^ nums[i], range(len(nums)), len(nums))

    # 137
    def singleNumber(self, nums: List[int]) -> int:
        ones, zeros = 0, ~0
        for num in nums:
            zeros = (zeros ^ num) & ~ones
            ones = (ones ^ num) & ~zeros
        return ones

    # 260
    def singleNumber(self, nums: List[int]) -> int:
        # TODO
        pass

    def reverseBits(self, n: int) -> int:
        ans = 0
        digitNum = 32
        while digitNum > 0:
            ans <<= 1
            ans += n % 2
            n >>= 1
            digitNum -= 1
        return ans
