import math
from typing import List, Optional

class Solution:
  # 215*
  # quick select O(n) 
  def findKthLargest(self, nums: List[int], k: int) -> int:
    def partition(nums, l, h):
      pivotIndex, smallBoundary = l, h-1
      while pivotIndex < smallBoundary:
        if nums[pivotIndex] < nums[pivotIndex+1]:
          nums[pivotIndex], nums[pivotIndex+1] = nums[pivotIndex+1], nums[pivotIndex]
          pivotIndex += 1
        else:
          nums[pivotIndex+1], nums[smallBoundary] = nums[smallBoundary], nums[pivotIndex+1]
          smallBoundary -= 1
      if pivotIndex == k - 1:
        return nums[pivotIndex]
      elif pivotIndex < k - 1:
        return partition(nums, pivotIndex+1, h)
      else:
        return partition(nums, l, pivotIndex)

    l, h = 0, len(nums)
    return partition(nums, l, h)  
  
  # 347*
  # bucket sort O(n)
  def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    countDict = {}
    bucketArray = [[] for i in range(len(nums)+1)]
    for num in nums:
      countDict[num] = 1 + countDict.get(num, 0)
    for num, count in countDict.items():
      bucketArray[count].append(num)
    
    result = []
    i = len(bucketArray) - 1
    while len(result) < k:
      result += bucketArray[i]
      i-=1
    return result

  #451
  def frequencySort(self, s: str) -> str:
    countDict = {}
    bucketArray = ["" for i in range(len(s)+1)]
    for c in s:
      countDict[c] = 1 + countDict.get(c, 0)
    for c, count in countDict.items():
      bucketArray[count] += c*count
    
    result = ""
    for i in range(len(bucketArray) - 1, 0, -1):
      result += bucketArray[i]
    return result

  #75* 荷兰国旗问题
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
        nums[i], nums[twoLowerBound-1] = nums[twoLowerBound-1], nums[i]
        twoLowerBound -= 1
