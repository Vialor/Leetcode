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
        stack1, stack2 = [], []
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next

        result = ListNode()
        carry = 0
        while stack1 or stack2 or carry:
            stack1Val = stack1.pop() if stack1 else 0
            stack2Val = stack2.pop() if stack2 else 0
            newVal = stack1Val + stack2Val + carry
            if newVal >= 10:
                newVal -= 10
                carry = 1
            else:
                carry = 0
            newNode = ListNode(val=newVal, next=result.next)
            result.next = newNode
        return result.next

    # 234
    # time: O(N), space: O(1)
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        if not slow.next:
            return True
        if not slow.next.next:
            return head.val == slow.next.val
        halfLength = 1
        tail, tailNext = slow.next, slow.next.next
        while tailNext:
            halfLength += 1
            temp = tailNext.next
            tailNext.next = tail
            tail, tailNext = tailNext, temp

        for i in range(halfLength):
            if head.val != tail.val:
                return False
            head, tail = head.next, tail.next
        return True
