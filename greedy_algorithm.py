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
  def maxProfit(self, prices: List[int]) -> int:
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit