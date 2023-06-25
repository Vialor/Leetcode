from typing import List


class Solution:
    # 1351
    def countNegatives(self, grid: List[List[int]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])
        i, j = 0, n - 1
        negToRight = 0
        for i in range(m):
            while j >= 0 and grid[i][j] < 0:
                negToRight += 1
                j -= 1
            ans += negToRight
        return ans

    # 57
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        ans = []
        n = len(intervals)
        i = 0
        while i < n and intervals[i][1] < newInterval[0]:
            ans.append(intervals[i])
            i += 1
        if i < n:
            newInterval[0] = min(intervals[i][0], newInterval[0])
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1
        ans.append(newInterval)
        while i < n:
            ans.append(intervals[i])
            i += 1
        return ans
