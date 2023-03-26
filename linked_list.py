from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 19
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    while n > 0:
        n -= 1
    return
