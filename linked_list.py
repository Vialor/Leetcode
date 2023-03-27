from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 19*
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        pre = ListNode(next=head)
        fast, slow = pre, pre
        while n >= 0:
            fast = fast.next
            n -= 1
        while fast:
            fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return slow.next

    # 24*
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        preHead = ListNode(next=head)
        last = preHead
        while last.next and last.next.next:
            # define related nodes
            nodeA = last.next
            nodeB = nodeA.next
            nextNode = nodeB.next
            # swap nodes
            last.next = nodeB
            nodeB.next = nodeA
            nodeA.next = nextNode
            # next iter
            last = nodeA
        return preHead.next

    # 445
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        pass
