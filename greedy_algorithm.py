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
    intervals.sort(key=lambda l : l[1])
    curEnd = intervals[0][1]
    count = 0
    for i in intervals[1:]:
      if i[0] < curEnd:
        count += 1
      else:
        curEnd = i[1]
    return count

  # 452
  def findMinArrowShots(self, points: List[List[int]]) -> int:
    pass