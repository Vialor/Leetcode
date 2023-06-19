import collections
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

    # 150
    def evalRPN(self, tokens: List[str]) -> int:
        item = tokens.pop()
        if item not in "+-*/":
            return int(item)
        secondOperand = self.evalRPN(tokens)
        firstOperand = self.evalRPN(tokens)
        match item:
            case "+":
                return firstOperand + secondOperand
            case "-":
                return firstOperand - secondOperand
            case "*":
                return firstOperand * secondOperand
            case "/":
                return int(firstOperand / secondOperand)

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
    # 84*
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.append(0)
        minStack = []
        ans = 0
        for i, height in enumerate(heights):
            while minStack and heights[minStack[-1]] > height:
                popIndex = minStack.pop()
                left = minStack[-1] if minStack else -1
                ans = max(ans, heights[popIndex] * (i - left - 1))
            minStack.append(i)
        return ans

    # 85*
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        ans = 0
        heights = [0] * n
        for i in range(m):
            heights = [heights[j] + 1 if matrix[i][j] == "1" else 0 for j in range(n)]
            heights.append(0)
            minStack = []
            for j, height in enumerate(heights):
                while minStack and heights[minStack[-1]] > height:
                    popIndex = minStack.pop()
                    left = minStack[-1] if minStack else -1
                    ans = max(ans, heights[popIndex] * (j - left - 1))
                minStack.append(j)
            heights.pop()
        return ans

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

    # monotonic deque
    # 239*
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = collections.deque()
        ans = []
        for i in range(len(nums)):
            if i - k >= 0 and queue[0] == i - k:
                queue.popleft()
            while queue and nums[queue[-1]] < nums[i]:
                queue.pop()
            queue.append(i)
            ans.append(nums[queue[0]])
        return ans[k - 1 :]

    # 862*
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        preSums = [0]
        for num in nums:
            preSums.append(num + preSums[-1])
        queue = collections.deque([0])
        shortest = float("inf")
        for i in range(1, len(preSums)):
            while queue and preSums[queue[-1]] >= preSums[i]:
                queue.pop()
            queue.append(i)
            while queue and preSums[queue[-1]] - preSums[queue[0]] >= k:
                shortest = min(shortest, queue[-1] - queue[0])
                queue.popleft()
        return -1 if shortest == float("inf") else shortest

    # 581
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        start, end = -1, -2
        curMax = float("-inf")
        for i in range(len(nums)):
            if nums[i] < curMax:
                end = i
            curMax = max(nums[i], curMax)
        curMin = float("inf")
        for i in reversed(range(len(nums))):
            if nums[i] > curMin:
                start = i
            curMin = min(nums[i], curMin)
        return end - start + 1

    # https://leetcode.com/problems/trapping-rain-water/ 42 with stack
    # 42* TODO
    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0
        for i in range(len(height)):
            stack.append(height[i])


# https://leetcode.com/problems/subarrays-with-k-different-integers/
# https://leetcode.com/problems/online-stock-span/
# https://leetcode.com/problems/sum-of-subarray-minimums/
