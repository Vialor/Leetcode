import collections
import math
from typing import List, Optional


class Solution:
    # 455
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        gIndex = 0
        sIndex = 0
        contentChildren = 0
        while gIndex < len(g) and sIndex < len(s):
            if g[gIndex] <= s[sIndex]:
                contentChildren += 1
                gIndex += 1
            sIndex += 1
        return contentChildren

    # 435*
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        curEnd = intervals[0][1]
        count = 0
        for i in intervals[1:]:
            if curEnd <= i[0]:
                curEnd = i[1]
            else:
                count += 1
        return count

    # 452
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[1])
        curEnd = points[0][1]
        count = 1
        for i in points[1:]:
            if curEnd < i[0]:
                curEnd = i[1]
                count += 1
        return count

    # 646
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort(key=lambda x: x[1])
        curEnd = pairs[0][1]
        count = 1
        for left, right in pairs[1:]:
            if left > curEnd:
                curEnd = right
                count += 1
        return count

    # 406
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda l: (-l[0], l[1]))
        result = []
        for p in people:
            result.insert(p[1], p)
        return result

    # 121
    def maxProfit121(self, prices: List[int]) -> int:
        lowValue = prices[0]
        profit = 0
        for price in prices[1:]:
            lowValue = min(price, lowValue)
            profit = max(price - lowValue, profit)
        return profit

    # 122
    def maxProfit122(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]
        return profit

    # 605
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0
        for i in range(len(flowerbed)):
            if flowerbed[i] == 1:
                continue
            if (i - 1 < 0 or flowerbed[i - 1] == 0) and (i + 1 >= len(flowerbed) or flowerbed[i + 1] == 0):
                flowerbed[i] = 1
                count += 1
        return count >= n

    # 392
    def isSubsequence(self, s: str, t: str) -> bool:
        sIndex = 0
        for c in t:
            if sIndex >= len(s):
                return True
            if s[sIndex] == c:
                sIndex += 1
        return sIndex == len(s)

    # 763
    def partitionLabels(self, s: str) -> List[int]:
        if len(s) == 0:
            return []
        lastOccurences = {s[i]: i for i in range(len(s))}
        i, end = 0, 1
        while i < end:
            end = max(end, lastOccurences[s[i]] + 1)
            i += 1
        return [end] + self.partitionLabels(s[end:])

    # 55
    def canJump(self, nums: List[int]) -> bool:
        smallestJumpableIndex = len(nums) - 1
        for i in reversed(range(len(nums) - 1)):
            if i + nums[i] >= smallestJumpableIndex:
                smallestJumpableIndex = i
        return smallestJumpableIndex == 0

    # 45
    def jump(self, nums: List[int]) -> int:
        stepCount = 0
        maxNextJump = 0
        curMax = 0
        for i in range(len(nums) - 1):
            maxNextJump = max(maxNextJump, nums[i] + i)
            if i == curMax:
                curMax = maxNextJump
                stepCount += 1
        return stepCount

    # 134
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        startIndex, stationNum = 0, len(gas)
        balance = 0
        for i in range(stationNum):
            balance = balance + gas[i] - cost[i]
            if balance < 0:
                balance = 0
                startIndex = i + 1
        return startIndex

    # 621
    # the greedy way, hard to think of, using queue is better
    def leastInterval(self, tasks: List[str], n: int) -> int:
        taskCounter = collections.Counter(tasks)
        maxFreq, maxCount = 0, 0
        for frequency in taskCounter.values():
            if frequency > maxFreq:
                maxFreq = frequency
                maxCount = 1
            elif frequency == maxFreq:
                maxCount += 1
        emptySlots = (maxFreq - 1) * (n - maxCount + 1)
        restTasks = len(tasks) - maxFreq * maxCount
        idles = max(0, emptySlots - restTasks)
        return idles + len(tasks)

    # 678
    def checkValidString(self, s: str) -> bool:
        leftCountMin, leftCountMax = 0, 0
        for c in s:
            if c == "(":
                leftCountMin += 1
                leftCountMax += 1
            elif c == ")":
                leftCountMin -= 1
                leftCountMax -= 1
            else:
                leftCountMin -= 1
                leftCountMax += 1
            if leftCountMax < 0:
                return False
            leftCountMin = max(0, leftCountMin)
        return leftCountMin == 0
