from functools import wraps
from typing import List, Optional
from tree import Solution


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def createLinkedList(array: List) -> Optional[ListNode]:
    if len(array) == 0:
        return None
    return ListNode(val=array[0], next=createLinkedList(array[1:]))

def createBinaryTree(array: List) -> Optional[TreeNode]:
    if len(array) == 0:
        return None
    mid = len(array)
    return TreeNode(val=array[0], next=createBinaryTree(array[1:]))

# from _ import Solution
# solution = Solution()
# solution.func()
