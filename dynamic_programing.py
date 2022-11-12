import functools
import math
from typing import List, Optional

class Solution:
  # FIB
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

  # PATH
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

  # INTERVAL
  # 413*
  def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    dp = [0, 0]
    for i in range(2, len(nums)):
      if 2*nums[i-1] == nums[i] + nums[i-2]:
        dp.append(dp[i-1] + 1)
      else:
        dp.append(0)
    return sum(dp)

  # INT BREAKDOWN
  # 343*
  def integerBreak(self, n: int) -> int:
    if n == 2: return 1
    if n == 3: return 2
    dp = [0, 1, 2, 3]
    for i in range(4, n + 1):
      dp.append(0)
      for j in range(1, i // 2 + 1):
        dp[i] = max(dp[j] * dp[i-j], dp[i])
    return dp[n]

  # 91
  def numDecodings(self, s: str) -> int:
    dp = [] # dp[k]: numDecodings(s[:k])
    # 0
    dp.append(1)
    # 1
    dp.append(0 if s[0] == '0' else 1)
    if len(s) == 1: return dp[1]
    # more than 1
    for i in range(2, len(s) + 1):
        lastTwo = int(s[i-2:i])
        lastOne = int(s[i-1])
        result = 0
        if 10 <= lastTwo <= 26: result += dp[i-2]
        if 1 <= lastOne <= 9: result += dp[i-1]
        dp.append(result)
    return dp[len(s)]

  # SEQUENCE
  # 300*
  def lengthOfLIS(self, nums: List[int]) -> int:
    LIS = [1] * len(nums) # LIS[k]: length of LIS that ends at kth index of nums
    for i in range(1, len(nums)):
      for j in range(i):
        if nums[j] < nums[i]:
          LIS[i] = max(LIS[i], LIS[j] + 1)
    return max(LIS)

  # 1143*
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    for i in range(1, m+1):
      for j in range(1, n+1):
        if text1[i-1] == text2[j-1]:
          dp[i][j] = dp[i-1][j-1] + 1
        else:
          dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
  
  # 376*
  def wiggleMaxLength(self, nums: List[int]) -> int:
    up, down = 1, 1 # up: length of wiggle subsequence that ends with an up-trend
    for i in range(1,len(nums)):
        if (nums[i] > nums[i - 1]):
            up = down + 1
        elif (nums[i] < nums[i - 1]):
            down = up + 1
    return max(up, down)

  # 0-1 backpack
  # 416* 2D
  # def canPartition(self, nums: List[int]) -> bool:
  #   total = sum(nums)
  #   if total % 2 == 1: return False
  #   targetAmount = total // 2
  #   # dp: [capacity][number of items considered]
  #   dp = [[0 for i in range(len(nums) + 1)] for c in range(targetAmount + 1)]
    
  #   for c in range(1, targetAmount + 1):
  #       for i in range(1, len(nums) + 1):
  #           curValue = nums[i-1]
  #           if c < curValue: dp[c][i] = dp[c][i-1]
  #           else:
  #               dp[c][i] = max(dp[c][i-1], curValue + dp[c-curValue][i-1])
  #   return dp[targetAmount][len(nums)] == targetAmount
  # 416* 1D
  def canPartition(self, nums: List[int]) -> bool:
    total = sum(nums)
    if total % 2:
      return False
    target = total // 2
    # dp[t]: if nums can be partitioned to reach sum t
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
      for t in range(target, num - 1, -1):
        dp[t] = dp[t] or dp[t - num]
    return dp[target]
  
  # 494
  def findTargetSumWays(self, nums: List[int], target: int) -> int:
    # sum(P) - sum(N) = target, sum(P) + sum(N) = sum(nums)
    # sum(P) = (target + sum(nums)) // 2
    if (target + sum(nums)) % 2: return 0
    if target + sum(nums) < 0: return 0
    targetPositiveSum = (target + sum(nums)) // 2
    # dp[t]: number of ways to form P to reach t
    dp = [1] + [0] * (targetPositiveSum)

    for num in nums:
      temp = dp[:]
      for t in range(num, targetPositiveSum+1):
        temp[t] += dp[t-num]
      dp = temp
    return dp[targetPositiveSum]
