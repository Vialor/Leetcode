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

    # 875*
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

    # 1011
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def canShip(capacity: int) -> bool:
            dayCount = 1
            carry = 0
            for weight in weights:
                if weight > capacity:
                    return False
                if carry + weight > capacity:
                    dayCount += 1
                    carry = 0
                carry += weight
                if dayCount > days:
                    return False
            return True

        low, high = 0, sum(weights)
        while low < high:
            mid = (low + high) // 2
            if canShip(mid):
                high = mid
            else:
                low = mid + 1
        return low

    # 410
    def splitArray(self, nums: List[int], k: int) -> int:
        preSums = [0]
        for num in nums:
            preSums.append(preSums[-1] + num)

        low, high = 0, preSums[-1]
        while low < high:
            mid = (low + high) // 2

            arrayCount = 0
            start, end = 0, 1
            while end < len(preSums) and arrayCount < k:
                while end < len(preSums) and preSums[end] - preSums[start] <= mid:
                    end += 1
                start = end - 1
                arrayCount += 1

            if end < len(preSums):
                low = mid + 1
            else:
                high = mid
        return low

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

    # 4
    def findMedianSortedArrays(self, A: List[int], B: List[int]) -> float:
        if len(A) > len(B):
            A, B = B, A
        AL, BL = len(A), len(B)
        ijSum = (AL + BL + 1) // 2
        low, high = 0, len(A)
        while True:
            i = (low + high) // 2
            j = ijSum - i
            if i > 0 and j < BL and A[i - 1] > B[j]:
                high = i - 1
            elif j > 0 and i < AL and B[j - 1] > A[i]:
                low = i + 1
            else:
                leftMost = A[i - 1] if i > 0 else float("-inf")
                if j > 0:
                    leftMost = max(leftMost, B[j - 1])
                rightMin = A[i] if i < AL else float("inf")
                if j < BL:
                    rightMin = min(rightMin, B[j])
                if (AL + BL) % 2 == 1:
                    return leftMost
                else:
                    return (leftMost + rightMin) / 2
