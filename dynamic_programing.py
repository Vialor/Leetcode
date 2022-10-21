import functools
import math
from typing import List, Optional

class Solution:
  # 70
  def climbStairs(self, n: int) -> int:
    dp = [0, 1, 2] # dp[k]: methods to climb k
    for i in range(3, n+1):
      dp.append(dp[i-1] + dp[i-2])
    return dp[n]

  # 198
  def rob1(self, nums: List[int]) -> int:
    if len(nums) == 1: return nums[0]
    dp = [nums[0], max(nums[0], nums[1])] # dp[k]: largest loot of nums[:k+1]
    for n in nums[2:]:
      dp.append(max(dp[-1], dp[-2]+n))
    return dp[-1]

  # 213*
  def rob2(self, nums: List[int]) -> int:
    if len(nums) == 1: return nums[0]
    return max(self.rob1(nums[1:]), self.rob1(nums[:-1]))

  # 64*
  def minPathSum(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    for i in range(m):
      for j in range(n):
        if i == 0 and j == 0: continue
        if i == 0: grid[i][j] += grid[i][j-1]
        elif j == 0: grid[i][j] += grid[i-1][j]
        else: grid[i][j] += min(grid[i][j-1], grid[i-1][j])
    return grid[m-1][n-1]

  # 62
  def uniquePaths(self, m: int, n: int) -> int:
    # C(c1, c2)
    c1 = m + n - 2
    c2 = min(m, n) - 1
    
    contMult = lambda p, q : functools.reduce(lambda a, b:a * b, range(p, q), 1)
    return int(contMult(c1 - c2 + 1, c1 + 1)/contMult(1, c2 + 1))

  # 300*
  def lengthOfLIS(self, nums: List[int]) -> int:
    LIS = [1] * len(nums) # LIS[k]: length of LIS that ends at kth index of nums
    for i in range(1, len(nums)):
      for j in range(i):
        if nums[j] < nums[i]:
          LIS[i] = max(LIS[i], LIS[j] + 1)
    return max(LIS)    

  # 416
  def canPartition(self, nums: List[int]) -> bool:
    total = sum(nums)
    if total % 2 == 1: return False
    targetAmount = total / 2