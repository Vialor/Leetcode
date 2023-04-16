import math
from typing import List, Optional


class Solution:
    # 167*
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        a, b = 1, len(numbers)
        while numbers[a - 1] + numbers[b - 1] != target:
            if numbers[a - 1] + numbers[b - 1] < target:
                a += 1
            else:
                b -= 1
        return [a, b]

    # 633
    def judgeSquareSum(self, c: int) -> bool:
        a, b = 0, math.floor(math.sqrt(c))
        while a <= b:
            sum = a**2 + b**2
            if sum < c:
                a += 1
            elif sum > c:
                b -= 1
            else:
                return True
        return False

    # 345
    def reverseVowels(self, s: str) -> str:
        vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
        result = list(s)
        a, b = 0, len(s) - 1
        while a <= b:
            if result[a] not in vowels:
                a += 1
            elif result[b] not in vowels:
                b -= 1
            else:
                result[a], result[b] = result[b], result[a]
                a += 1
                b -= 1
        return "".join(result)

    # 680
    def validPalindrome(self, s: str) -> bool:
        a, b = 0, len(s) - 1
        while a < b:
            if s[a] != s[b]:
                s1 = s[:a] + s[a + 1 :]
                s2 = s[:b] + s[b + 1 :]
                return s1 == s1[::-1] or s2 == s2[::-1]
            a += 1
            b -= 1
        return True

    # 88
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        index1, index2, indexMerge = m - 1, n - 1, m + n - 1
        while indexMerge >= 0:
            if index1 < 0 or (index2 >= 0 and nums1[index1] < nums2[index2]):
                nums1[indexMerge] = nums2[index2]
                index2 -= 1
            else:
                nums1[indexMerge] = nums1[index1]
                index1 -= 1
            indexMerge -= 1

    # 141*
    class ListNode:
        def __init__(self):
            self.next = None

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast, slow = head.next, head
        while slow != None and fast != None and fast.next != None:
            if fast == slow:
                return True
            fast = fast.next.next
            slow = slow.next
        return False

    # 524
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        def stringMatched(s1: str, s2: str) -> bool:
            a, b = 0, 0
            while a < len(s1) and b < len(s2):
                if s1[a] == s2[b]:
                    b += 1
                a += 1
            return b == len(s2)

        longestWord, longestWordLength = "", 0
        for word in dictionary:
            if stringMatched(s, word) and (
                len(word) > longestWordLength or (len(word) == longestWordLength and word < longestWord)
            ):
                longestWord, longestWordLength = word, len(word)
        return longestWord

    # 11*
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        mostVolumn = 0
        while left < right:
            leftHeight, rightHeight = height[left], height[right]
            if leftHeight < rightHeight:
                mostVolumn = max(mostVolumn, leftHeight * (right - left))
                left += 1
            else:
                mostVolumn = max(mostVolumn, rightHeight * (right - left))
                right -= 1
        return mostVolumn

    # 15*
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []

        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            a = i + 1
            b = len(nums) - 1
            while a < b:
                if nums[i] + nums[a] + nums[b] < 0:
                    a += 1
                elif nums[i] + nums[a] + nums[b] > 0:
                    b -= 1
                else:
                    result.append([nums[i], nums[a], nums[b]])
                    while a < b and nums[a] == nums[a + 1]:
                        a += 1
                    while a < b and nums[b] == nums[b - 1]:
                        b -= 1
                    a += 1
                    b -= 1
        return result

    # 26
    def removeDuplicates(self, nums: List[int]) -> int:
        a, b = 0, 0
        while b < len(nums):
            while b + 1 < len(nums) and nums[b] == nums[b + 1]:
                b += 1
            nums[a] = nums[b]
            a += 1
            b += 1
        nums = nums[:a]
        return a

    # 18
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        length = len(nums)
        for a in range(length):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            for b in range(a + 1, length):
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                i, j = b + 1, length - 1
                while i < j:
                    if nums[a] + nums[b] + nums[i] + nums[j] < target:
                        i += 1
                    elif nums[a] + nums[b] + nums[i] + nums[j] > target:
                        j -= 1
                    else:
                        result.append([nums[a], nums[b], nums[i], nums[j]])
                        while i < j and nums[i] == nums[i + 1]:
                            i += 1
                        i += 1
                        while i < j and nums[j] == nums[j - 1]:
                            j -= 1
                        j -= 1
        return result
