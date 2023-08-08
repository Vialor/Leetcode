from functools import reduce
import heapq
import random
from typing import List, Optional


class Solution:
    # 215*
    # quick select O(n)
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(nums, l, h):
            pivotIndex, smallBoundary = l, h - 1
            while pivotIndex < smallBoundary:
                if nums[pivotIndex] < nums[pivotIndex + 1]:
                    nums[pivotIndex], nums[pivotIndex + 1] = nums[pivotIndex + 1], nums[pivotIndex]
                    pivotIndex += 1
                else:
                    nums[pivotIndex + 1], nums[smallBoundary] = nums[smallBoundary], nums[pivotIndex + 1]
                    smallBoundary -= 1
            if pivotIndex == k - 1:
                return nums[pivotIndex]
            elif pivotIndex < k - 1:
                return partition(nums, pivotIndex + 1, h)
            else:
                return partition(nums, l, pivotIndex)

        l, h = 0, len(nums)
        return partition(nums, l, h)

    # 973
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        def quickSelect(low, high):
            if low == high:
                return
            pivot = random.randint(low, high)
            innerLow, innerHigh = low, high
            countLessThan = low
            while innerLow < innerHigh:
                if points[innerLow] < pivot:
                    countLessThan += 1
                    innerLow += 1
                else:
                    points[innerHigh], points[innerLow] = points[innerLow], points[innerHigh]
                    high -= 1
            if countLessThan > k:
                quickSelect(innerLow, high)
            elif countLessThan < k:
                quickSelect(low, innerLow)
            else:
                return

        quickSelect(0, len(points) - 1)
        return points[:k]

    # 462
    def minMoves2(self, nums: List[int]) -> int:
        def quickSelect(low, high) -> int:
            pivot = random.randint(low, high)
            nums[high], nums[pivot] = nums[pivot], nums[high]
            curInd = low
            for i in range(low, high):
                if nums[i] < nums[high]:
                    nums[curInd], nums[i] = nums[i], nums[curInd]
                    curInd += 1
            nums[high], nums[curInd] = nums[curInd], nums[high]

            if curInd == midLen:
                return nums[curInd]
            elif curInd > midLen:
                return quickSelect(low, curInd - 1)
            else:
                return quickSelect(curInd + 1, high)

        midLen = len(nums) // 2
        medium = quickSelect(0, len(nums) - 1)
        ans = reduce(lambda last, num: last + abs(num - medium), nums, 0)
        return ans

    # 347*
    # bucket sort O(n)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        countDict = {}
        bucketArray = [[] for i in range(len(nums) + 1)]
        for num in nums:
            countDict[num] = 1 + countDict.get(num, 0)
        for num, count in countDict.items():
            bucketArray[count].append(num)

        result = []
        i = len(bucketArray) - 1
        while len(result) < k:
            result += bucketArray[i]
            i -= 1
        return result

    # 451
    def frequencySort(self, s: str) -> str:
        countDict = {}
        bucketArray = ["" for i in range(len(s) + 1)]
        for c in s:
            countDict[c] = 1 + countDict.get(c, 0)
        for c, count in countDict.items():
            bucketArray[count] += c * count

        result = ""
        for i in range(len(bucketArray) - 1, 0, -1):
            result += bucketArray[i]
        return result

    # 75* 荷兰国旗问题
    def sortColors(self, nums: List[int]) -> None:
        zeroUpperBound, i, twoLowerBound = 0, 0, len(nums)
        # range(lowerBound, UpperBound)
        while i < twoLowerBound:
            if nums[i] == 0:
                nums[zeroUpperBound], nums[i] = nums[i], nums[zeroUpperBound]
                zeroUpperBound += 1
                i += 1
            elif nums[i] == 1:
                i += 1
            else:
                nums[i], nums[twoLowerBound - 1] = nums[twoLowerBound - 1], nums[i]
                twoLowerBound -= 1

    # 324*
    def wiggleSort(self, nums: List[int]) -> None:
        # or quick select median
        median = heapq.nlargest((len(nums) // 2) + 1, nums)[-1]

        def indexMapping(i):
            return (1 + 2 * i) % (len(nums) | 1)

        # three way partitioning
        low, mid, high = 0, 0, len(nums) - 1
        while mid <= high:
            if nums[indexMapping(mid)] == median:
                mid += 1
            elif nums[indexMapping(mid)] > median:
                nums[indexMapping(mid)], nums[indexMapping(low)] = nums[indexMapping(low)], nums[indexMapping(mid)]
                mid += 1
                low += 1
            else:
                nums[indexMapping(mid)], nums[indexMapping(high)] = nums[indexMapping(high)], nums[indexMapping(mid)]
                high -= 1

    # 315*
    def countSmaller(self, nums: List[int]) -> List[int]:
        ans = [0] * len(nums)
        sortedIndices = list(range(len(nums)))

        def conquer(start, end):
            if end - start <= 1:
                return

            middle = (start + end) // 2
            conquer(start, middle)
            conquer(middle, end)

            nonlocal sortedIndices
            localSortedIndices = []
            rightSmallCount = 0
            i, j = start, middle
            while i < middle or j < end:
                if j < end and (i >= middle or i < middle and nums[sortedIndices[i]] > nums[sortedIndices[j]]):
                    localSortedIndices.append(sortedIndices[j])
                    rightSmallCount += 1
                    j += 1
                else:
                    localSortedIndices.append(sortedIndices[i])
                    ans[sortedIndices[i]] += rightSmallCount
                    i += 1
            sortedIndices[start:end] = localSortedIndices

        conquer(0, len(nums))
        return ans
