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
            mid = (high + low + 1) // 2
            if nums[mid] <= target:
                low = mid
            else:
                high = mid - 1
        end = low
        return [start, end]

    # 875
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        low, high = 1, max(piles)
        minSpead = high
        while low <= high:
            mid = (high + low) // 2
            eatTime = 0
            for pile in piles:
                eatTime += pile // mid
                if pile % mid != 0:
                    eatTime += 1
            if eatTime <= h:
                minSpead = min(minSpead, mid)
                high = mid - 1
            else:
                low = mid + 1
        return minSpead

    # 74*
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        low, high = 0, m * n - 1
        while low <= high:
            mid = (low + high) // 2
            row, col = mid // n, mid % n
            if matrix[row][col] < target:
                low = mid + 1
            elif matrix[row][col] > target:
                high = mid - 1
            else:
                return True
        return False

    # 33
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1
        targetOnLeft = target >= nums[0]
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            midOnLeft = nums[mid] >= nums[0]
            if targetOnLeft and not midOnLeft:
                high = mid - 1
            elif not targetOnLeft and midOnLeft:
                low = mid + 1
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    # 981
    class TimeMap:
        def __init__(self):
            self.keyMap = {}
            # key: [[timestamps], [values]]

        def set(self, key: str, value: str, timestamp: int) -> None:
            if key in self.keyMap:
                self.keyMap[key][0].append(timestamp)
                self.keyMap[key][1].append(value)
            else:
                self.keyMap[key] = [[timestamp], [value]]

        def get(self, key: str, timestamp: int) -> str:
            if key in self.keyMap:
                timestamps, values = self.keyMap[key]
                low, high = 0, len(timestamps) - 1
                while low < high:
                    mid = (low + high + 1) // 2
                    if timestamps[mid] <= timestamp:
                        low = mid
                    else:
                        high = mid - 1
                return values[low] if timestamps[low] <= timestamp else ""
            else:
                return ""
