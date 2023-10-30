from typing import List, Optional

from test_tools import TestTools


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 206*
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        newHead = None
        while head:
            head.next, newHead, head = newHead, head, head.next
        return newHead

    # 92*
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        pre = dummy
        for _ in range(left - 1):
            pre = pre.next
        cur = pre.next
        for _ in range(right - left):
            temp = cur.next
            cur.next = temp.next
            temp.next = pre.next
            pre.next = temp
        return dummy.next

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

    # 25
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        preNode, postNode = ListNode(next=head), head
        newHead = preNode
        while True:
            for i in range(k):
                if postNode is None:
                    return newHead.next
                postNode = postNode.next

            nodeA = postNode
            nodeB = preNode.next
            nextPreNode = preNode.next
            nodeC = nodeB.next
            while nodeB != postNode:
                nodeB.next = nodeA
                nodeA = nodeB
                nodeB = nodeC
                if nodeC:
                    nodeC = nodeC.next
            preNode.next = nodeA
            preNode = nextPreNode

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
    # time: O(n), space: O(1)
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

    # 725
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        length = 0
        cur = head
        while cur:
            length += 1
            cur = cur.next

        shortLength = length // k
        longLength = shortLength + 1
        longNumber = length % k

        def isBreakPoint(n: int) -> bool:
            if n <= longLength * longNumber:
                return n % longLength == 0
            else:
                return (n - longLength * longNumber) % shortLength == 0

        result = []
        count = 0
        cur, curNext = ListNode(next=head), head
        while curNext:
            if isBreakPoint(count):
                cur.next = None
                result.append(curNext)
            count += 1
            cur, curNext = curNext, curNext.next
        while len(result) < k:
            result.append(None)
        return result

    # 328
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        oddCur, evenCur = head, head.next
        evenHead = evenCur
        while oddCur.next and evenCur.next:
            oddCur.next = oddCur.next.next
            evenCur.next = evenCur.next.next
            oddCur = oddCur.next
            evenCur = evenCur.next
        oddCur.next = evenHead
        return head

    # 142
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return fast
        return None

    # 1721
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        fast, slow = dummy, dummy
        for _ in range(k - 1):
            fast = fast.next
        prea = fast
        a = fast.next
        fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        preb = slow
        b = slow.next

        tmp = b.next
        prea.next = b
        b.next = a.next
        preb.next = a
        a.next = tmp
        return dummy.next

    # 143*
    def reorderList(self, head: Optional[ListNode]) -> None:
        if head is None:
            return

        # divide the list into halves
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        insertHead = slow.next
        slow.next = None

        # reverse the inserting list
        anode = None
        bnode = insertHead
        while bnode:
            cnode = bnode.next
            bnode.next = anode
            anode = bnode
            bnode = cnode
        insertHead = anode

        # merge lists
        cur = head
        while insertHead:
            curNext = cur.next
            insertHeadNext = insertHead.next

            cur.next = insertHead
            insertHead.next = curNext

            cur = curNext
            insertHead = insertHeadNext


# 146
class DoublyLinkedNode:
    def __init__(self, key: int, val: int, prevNode=None, nextNode=None):
        self.key = key
        self.val = val
        self.prev = prevNode
        self.next = nextNode


class LRUCache:
    def __init__(self, capacity: int):
        initialNode = DoublyLinkedNode(-1, -1)
        self.capacity = capacity
        self.occupied = 0
        self.LRUListHead = initialNode
        self.LRUListTail = initialNode
        self.mapping = {}

    def moveToMostRecent(self, node: DoublyLinkedNode):
        if node == self.LRUListTail:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self.LRUListTail.next = node
        node.prev = self.LRUListTail
        node.next = None
        self.LRUListTail = node

    def get(self, key: int) -> int:
        if key not in self.mapping:
            return -1
        node = self.mapping[key]
        self.moveToMostRecent(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.mapping:
            node = self.mapping[key]
            self.moveToMostRecent(node)
            node.val = value
        else:
            if self.occupied >= self.capacity:
                del self.mapping[self.LRUListHead.next.key]
                self.LRUListHead.next = self.LRUListHead.next.next
                if self.LRUListHead.next:
                    self.LRUListHead.next.prev = self.LRUListHead
                else:
                    self.LRUListTail = self.LRUListHead
            else:
                self.occupied += 1
            node = DoublyLinkedNode(key, value)
            self.mapping[key] = node
            self.LRUListTail.next = node
            node.prev = self.LRUListTail
            self.LRUListTail = node

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

    # 86*
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        preLeft, preRight = ListNode(), ListNode()
        leftCur, rightCur = preLeft, preRight
        while head:
            if head.val < x:
                leftCur.next = head
                leftCur = leftCur.next
            else:
                rightCur.next = head
                rightCur = rightCur.next
            head = head.next

        leftCur.next = preRight.next
        rightCur.next = None
        return preLeft.next
