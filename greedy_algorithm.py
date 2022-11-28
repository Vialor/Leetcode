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
    if not intervals: return 0
    intervals.sort(key=lambda x : x[1])
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
    points.sort(key=lambda x : x[1])
    curEnd = points[0][1]
    count = 1
    for i in points[1:]:
      if curEnd < i[0]:
        curEnd = i[1]
        count += 1
    return count

  # 646
  def findLongestChain(self, pairs: List[List[int]]) -> int:
    pairs.sort(key=lambda x : x[1])
    curEnd = pairs[0][1]
    count = 1
    for left, right in pairs[1:]:
      if left > curEnd:
        curEnd = right
        count += 1
    return count

  # 406
  def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
    people.sort(key=lambda l : (-l[0], l[1]))
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
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit
  
  # 605
  def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    count = 0
    for i in range(len(flowerbed)):
      if flowerbed[i] == 1: continue
      if (i-1 < 0 or flowerbed[i-1] == 0) and (i+1 >= len(flowerbed) or flowerbed[i+1] == 0):
        flowerbed[i] = 1
        count += 1
    return count >= n

  # 392
  def isSubsequence(self, s: str, t: str) -> bool:
    sIndex = 0
    for c in t:
      if sIndex >= len(s): return True
      if s[sIndex] == c:
        sIndex += 1
    return sIndex == len(s)
  
  # 53
  def maxSubArray(self, nums: List[int]) -> int:
    curSum = 0
    maxSum = nums[0]
    for num in nums:
      curSum += num
      maxSum = max(curSum, maxSum)
      if curSum < 0: curSum = 0
    return maxSum

  # 763
  def partitionLabels(self, s: str) -> List[int]:
    if len(s) == 0: return [] 
    lastOccurences = { s[i]: i for i in range(len(s)) }
    i, end = 0, 1
    while i < end:
      end = max(end, lastOccurences[s[i]] + 1)
      i += 1
    return [end] + self.partitionLabels(s[end:])