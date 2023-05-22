import collections
import math
from typing import List, Optional


class Solution:
    # 232
    class MyQueue:
        def __init__(self):
            self.inStack = []
            self.outStack = []

        def push(self, x: int) -> None:
            self.inStack.append(x)

        def inStackToOutStack(self) -> None:
            while self.inStack:
                self.outStack.append(self.inStack.pop())

        def pop(self) -> int:
            if not self.outStack:
                self.inStackToOutStack()
            return self.outStack.pop()

        def peek(self) -> int:
            if not self.outStack:
                self.inStackToOutStack()
            return self.outStack[-1]

        def empty(self) -> bool:
            return not self.inStack and not self.outStack

    # 225
    class MyStack:
        def __init__(self):
            self.queue = collections.deque()

        def push(self, x: int) -> None:
            count = len(self.queue)
            self.queue.append(x)
            while count:
                self.queue.append(self.queue.popleft)
                count -= 1

        def pop(self) -> int:
            return self.queue.popleft()

        def top(self) -> int:
            return self.queue[0]

        def empty(self) -> bool:
            return not self.queue

    # 155
    class MinStack:
        def __init__(self):
            self.dataStack = []
            self.minStack = []

        def push(self, val: int) -> None:
            self.dataStack.append(val)
            minVal = min(self.minStack[-1], val) if self.minStack else val
            self.minStack.append(minVal)

        def pop(self) -> None:
            self.dataStack.pop()
            self.minStack.pop()

        def top(self) -> int:
            return self.dataStack[-1]

        def getMin(self) -> int:
            return self.minStack[-1]

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

    # 739*
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        result = [0] * len(temperatures)
        for i in reversed(range(len(temperatures))):
            while stack and temperatures[stack[-1]] <= temperatures[i]:
                stack.pop()
            if stack:
                result[i] = stack[-1] - i
            stack.append(i)
        return result

    # 503*
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        result = [-1] * len(nums)
        stack = []
        for i in reversed(range(len(nums) * 2 - 1)):
            while stack and stack[-1] <= nums[i % len(nums)]:
                stack.pop()
            if stack:
                result[i % len(nums)] = stack[-1]
            stack.append(nums[i % len(nums)])
        return result

    # 503 second method
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        last = None
        for n in nums:
            if last is None or n > last:
                stack.append(n)
                last = n
        stack.reverse()

        result = [-1] * len(nums)
        for i in reversed(range(len(nums))):
            while stack and stack[-1] <= nums[i]:
                stack.pop()
            if stack:
                result[i] = stack[-1]
            stack.append(nums[i])
        return result

    # 42* TODO
    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0
        for i in range(len(height)):
            stack.append(height[i])


# 84
# 85
