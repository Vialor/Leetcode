import functools
import math
from typing import List, Optional


class Solution:
    # 69*
    def mySqrt(self, x: int) -> int:
        low, high = 0, x
        while low < high:
            mid = (low + high) // 2 + 1
            if mid * mid > x:
                high = mid - 1
            else:
                low = mid
        return low

    # 744
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        low = 0
        high = len(letters) - 1
        while high != low:
            mid = (high + low) // 2
            if letters[mid] <= target:
                low = mid + 1
            else:
                high = mid
        return letters[low] if letters[low] > target else letters[0]

    # 540
    def singleNonDuplicate(self, nums: List[int]) -> int:
        low = 0
        high = len(nums) - 1
        while high > low:
            mid = (low + high) // 2
            if (mid % 2 == 0 and nums[mid + 1] == nums[mid]) or (mid % 2 == 1 and nums[mid - 1] == nums[mid]):
                low = mid + 1
            else:
                high = mid
        return nums[low]

    # 153
    def findMin(self, nums: List[int]) -> int:
        low = 0
        high = len(nums) - 1
        while high > low:
            mid = (low + high) // 2
            if nums[mid] < nums[-1]:
                high = mid
            else:
                low = mid + 1
        return nums[low]

    # 34*
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]

        low, high = 0, len(nums) - 1
        while low < high:
            mid = (high + low) // 2
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid
        start = low
        if nums[start] != target:
            return [-1, -1]

        low, high = 0, len(nums) - 1
        while low < high:
            mid = (high + low) // 2 + 1
            if nums[mid] <= target:
                low = mid
            else:
                high = mid - 1
        end = low
        return [start, end]
