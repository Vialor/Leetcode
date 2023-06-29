from collections import Counter
import collections
from functools import reduce, wraps
from typing import List, Optional
from two_pointers import Solution


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TestTools:
    def serializeLinkedList(self, head: Optional[ListNode]) -> List:
        ans = []
        while head:
            ans.append(head.val)
            head = head.next
        return ans

    def deserializeLinkedList(self, array: List) -> Optional[ListNode]:
        if len(array) == 0:
            return None
        return ListNode(val=array[0], next=self.deserializeLinkedList(array[1:]))

    def serializeTree(self, root):
        res = []
        BFSQueue = collections.deque([root])
        while BFSQueue:
            node = BFSQueue.popleft()
            if node:
                res.append(node.val)
                BFSQueue.append(node.left)
                BFSQueue.append(node.right)
            else:
                res.append(None)
        while res and res[-1] is None:
            res.pop()
        return res

    def deserializeTree(self, data):
        if len(data) == 0:
            return None
        i = 0
        head = TreeNode(val=data[i])
        BFSQueue = collections.deque([head])
        while i < len(data):
            node = BFSQueue.popleft()
            if node:
                i += 1
                node.left = TreeNode(val=data[i]) if i < len(data) and data[i] is not None else None
                i += 1
                node.right = TreeNode(val=data[i]) if i < len(data) and data[i] is not None else None
                BFSQueue.append(node.left)
                BFSQueue.append(node.right)
        return head
