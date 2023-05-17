import math
from typing import List, Optional


class Solution:
    # monotonic stack
    # 496
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        mapping = {}
        for i in range(len(nums2) - 1, -1, -1):
            while stack and stack[-1] < nums2[i]:
                stack.pop()
            mapping[nums2[i]] = stack[-1] if stack else -1
            stack.append(nums2[i])
        return [mapping[num] for num in nums1]

    # 42
    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0
        for i in range(len(height)):
            stack.append(height[i])


# 84
# 85
