import math
from typing import List, Optional

class Solution:
    # 167*
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        a, b = 1, len(numbers)
        while numbers[a-1] + numbers[b-1] != target:
            if numbers[a-1] + numbers[b-1] < target:
                a+=1
            else:
                b-=1
        return [a, b]

    # 633
    def judgeSquareSum(self, c: int) -> bool:
        a, b = 0, math.floor(math.sqrt(c))
        while a <= b:
            sum = a**2 + b**2
            if sum < c:
                a+=1
            elif sum > c:
                b-=1
            else:
                return True
        return False
    
    # 345
    def reverseVowels(self, s: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        result = list(s)
        a, b = 0, len(s)-1
        while a <= b:
            if result[a] not in vowels:
                a+=1
            elif result[b] not in vowels:
                b-=1
            else:
                result[a], result[b] = result[b], result[a]
                a+=1
                b-=1
        return "".join(result)
    
    # 680
    def validPalindrome(self, s: str) -> bool:
        a, b = 0, len(s)-1
        while a < b:
            if s[a] != s[b]:
                s1 = s[:a]+s[a+1:]
                s2 = s[:b]+s[b+1:]
                return s1 == s1[::-1] or s2 == s2[::-1]
            a+=1
            b-=1
        return True
    
    # 88
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        index1, index2, indexMerge = m-1, n-1, m+n-1
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
            if fast == slow: return True
            fast = fast.next.next
            slow = slow.next
        return False
    # food for thought:
    # why val is irrelavent? 
    # why we need slow speed?
    # why we choose 2?
    # what is the why the complexity is O(n)?

    # 524
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        # https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/description/
        pass

    # 11*
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        mostVolumn = 0
        while left < right:
            leftHeight, rightHeight = height[left], height[right]
            if leftHeight < rightHeight:
                mostVolumn = max(mostVolumn, leftHeight*(right-left))
                left += 1
            else:
                mostVolumn = max(mostVolumn, rightHeight*(right-left))
                right -= 1
        return mostVolumn

    # 15 26