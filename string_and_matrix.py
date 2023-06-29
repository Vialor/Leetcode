from functools import cache
import math
from typing import List, Optional


class Solution:
    # 696*
    def countBinarySubstrings(self, s: str) -> int:
        preLen, curLen = 0, 0
        count = 0
        for i in range(len(s)):
            curLen += 1
            if i + 1 >= len(s) or s[i] != s[i + 1]:
                count += min(preLen, curLen)
                preLen = curLen
                curLen = 0
        return count

    # 283*
    def moveZeroes(self, nums: List[int]) -> None:
        count = 0
        ind = 0
        for num in nums:
            if num == 0:
                count += 1
            else:
                nums[ind] = num
                ind += 1
        while ind < len(nums):
            nums[ind] = 0
            ind += 1

    # 566
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        if m * n != r * c:
            return mat
        result = [[0] * c for rind in range(r)]
        ind = 0
        for i in range(m):
            for j in range(n):
                result[ind // c][ind % c] = mat[i][j]
                ind += 1
        return result

    # 240
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        chigh, rlow = len(matrix[0]) - 1, 0
        while len(matrix) > rlow and chigh >= 0:
            if matrix[rlow][chigh] == target:
                return True
            if matrix[rlow][chigh] > target:
                chigh -= 1
            else:
                rlow += 1
        return False

    # 378*
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)

        def countNoMoreThan(t: int) -> int:  # O(n)
            count = 0
            i, j = n - 1, 0
            while i >= 0 and j < n:
                if matrix[i][j] > t:
                    i -= 1
                else:
                    count += i + 1
                    j += 1
            return count

        low, high = matrix[0][0], matrix[-1][-1]
        while low < high:
            mid = (low + high) // 2
            if countNoMoreThan(mid) < k:
                low = mid + 1
            else:
                high = mid
        return low

    # 287 nums is modified
    def findDuplicate(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            while nums[i] != i + 1 and nums[i] != nums[nums[i] - 1]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        return nums[-1]

    # 645*
    def findErrorNums(self, nums: List[int]) -> List[int]:  # time: O(n), space: O(1)
        # O(n) sort
        for i in range(len(nums)):
            while nums[i] != i + 1 and nums[i] != nums[nums[i] - 1]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        # find error nums
        for i in range(len(nums)):
            if nums[i] != i + 1:
                return [nums[i], i + 1]
        return [-1, -1]

    # 667
    def constructArray(self, n: int, k: int) -> List[int]:
        result = [i for i in range(1, n - k + 1)]
        operation = 1
        for i in range(k, 0, -1):
            result.append(result[-1] + operation * i)
            operation = -operation
        return result

    # 565
    def arrayNesting(self, nums: List[int]) -> int:
        maxCount = 0
        for i in range(len(nums)):
            count = 0
            while nums[i] != -1:
                nums[i], i = -1, nums[i]
                count += 1
            maxCount = max(count, maxCount)
        return maxCount

    # 769
    def maxChunksToSorted(self, arr: List[int]) -> int:
        maxSoFar = 0
        count = 0
        for i in range(len(arr)):
            maxSoFar = max(maxSoFar, arr[i])
            if maxSoFar <= i:
                count += 1
        return count

    # 169*
    # Boyer-Moore Majority Vote Algorithm
    def majorityElement(self, nums: List[int]) -> int:
        vote = 0
        candidate = nums[0]
        for num in nums:
            if vote <= 0:
                candidate = num
            vote = vote + 1 if candidate == num else vote - 1
        return candidate

    # 59
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for i in range(n)]
        directions, turn = ["right", "down", "left", "up"], 0
        i, j = 0, 0
        for num in range(1, n**2 + 1):
            matrix[i][j] = num
            match directions[turn % 4]:
                case "right":
                    j += 1
                    if j == n - turn // 4 - 1:
                        turn += 1
                case "down":
                    i += 1
                    if i == n - turn // 4 - 1:
                        turn += 1
                case "left":
                    j -= 1
                    if j == turn // 4:
                        turn += 1
                case "up":
                    i -= 1
                    if i == turn // 4 + 1:
                        turn += 1
        return matrix

    # 73
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        firstCol, firstRow = False, False
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    if i == 0:
                        firstRow = True
                    if j == 0:
                        firstCol = True
                    if i > 0 and j > 0:
                        matrix[i][0] = matrix[0][j] = 0

        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0
        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0

        if firstCol:
            for i in range(m):
                matrix[i][0] = 0
        if firstRow:
            for j in range(n):
                matrix[0][j] = 0

    # 10*
    @cache
    def isMatch(self, s: str, p: str) -> bool:
        if s == "" and p == "":
            return True
        followedByStar = len(p) > 1 and p[1] == "*"
        firstMatch = len(s) > 0 and len(p) > 0 and (p[0] == s[0] or p[0] == ".")
        if firstMatch and followedByStar:
            return self.isMatch(s[1:], p) or self.isMatch(s, p[2:])
        elif firstMatch and not followedByStar:
            return self.isMatch(s[1:], p[1:])
        elif not firstMatch and followedByStar:
            return self.isMatch(s, p[2:])
        return False
