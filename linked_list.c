#include <stdio.h>
#include <string.h>

struct ListNode {
    int val;
    struct ListNode *next;
};

// 160
struct ListNode *getIntersectionNode(struct ListNode *headA, struct ListNode *headB) {
  struct ListNode *a = headA, *b = headB;
  while (a != b) {
    a = a ? a -> next : headB;
    b = b ? b -> next : headA;
  }
  return a;
}

// 206
struct ListNode* reverseList(struct ListNode* head){
  struct ListNode *curNode = head, *nextNode = NULL, *lastNode = NULL;
  while (curNode) {
    nextNode = curNode->next;
    curNode->next = lastNode;
    lastNode = curNode;
    curNode = nextNode;
  }
  return lastNode;
}

// 21
struct ListNode* mergeTwoLists(struct ListNode* list1, struct ListNode* list2){
  if (!list1) return list2;
  if (!list2) return list1;
  struct ListNode* head = list1->val < list2->val ? list1 : list2;
  struct ListNode* cur = head;
  while (list1 || list2) {
    if (!list2 || list1 && list1->val < list2->val) {
      cur->next = list1;
      list1 = list1->next;
    } else {
      cur->next = list2;
      list2 = list2->next;
    }
    cur = cur->next;
  }
  return head;
}

struct ListNode* mergeTwoLists(struct ListNode* list1, struct ListNode* list2){
  if (!list1) return list2;
  if (!list2) return list1;
  if (list1->val < list2->val) {
    list1->next = mergeTwoLists(list1->next, list2);
    return list1;
  } else {
    list2->next = mergeTwoLists(list1, list2->next);
    return list2;
  }
}