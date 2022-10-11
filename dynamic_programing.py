import math
from typing import List, Optional

class Solution:
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