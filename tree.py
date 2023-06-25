from functools import cache
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # A. RECURSION
    # 104*
    # not dp-optimized
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    # 110
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def getHeight(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            leftHeight, rightHeight = getHeight(node.left), getHeight(node.right)
            if leftHeight == -1 or rightHeight == -1 or abs(leftHeight - rightHeight) > 1:
                return -1
            return max(leftHeight, rightHeight) + 1

        return getHeight(root) != -1

    # 543
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # (diameter of subtree, height of subtree)
        def DFS(node: Optional[TreeNode]):
            if not node:
                return (0, 0)
            leftDiameter, leftHeight = DFS(node.left)
            rightDiameter, rightHeight = DFS(node.right)
            return (max(leftDiameter, rightDiameter, leftHeight + rightHeight), max(leftHeight, rightHeight) + 1)

        return DFS(root)[0]

    # 226
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root

    # 617
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1
        result = root1
        result.val += root2.val
        result.left = self.mergeTrees(root1.left, root2.left)
        result.right = self.mergeTrees(root1.right, root2.right)
        return result

    # 112
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return targetSum == root.val
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)

    # 437
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if not root:
            return 0

        def pathSumWithRoot(curNode: Optional[TreeNode], targetSum: int) -> int:
            if not curNode:
                return 0
            return (
                pathSumWithRoot(curNode.left, targetSum - curNode.val)
                + pathSumWithRoot(curNode.right, targetSum - curNode.val)
                + (1 if curNode.val == targetSum else 0)
            )

        return (
            self.pathSum(root.left, targetSum)
            + self.pathSum(root.right, targetSum)
            + pathSumWithRoot(root.left, targetSum - root.val)
            + pathSumWithRoot(root.right, targetSum - root.val)
            + (1 if root.val == targetSum else 0)
        )

    # 572*
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSameTree(tree1, tree2):
            if not tree1 and not tree2:
                return True
            if not tree1 or not tree2:
                return False
            return (
                tree1.val == tree2.val and isSameTree(tree1.left, tree2.left) and isSameTree(tree1.right, tree2.right)
            )

        if not root:
            return subRoot is None
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot) or isSameTree(root, subRoot)

    # 101
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def isMirrorTree(tree1, tree2):
            if not tree1 and not tree2:
                return True
            if not tree1 or not tree2:
                return False
            return (
                tree1.val == tree2.val
                and isMirrorTree(tree1.left, tree2.right)
                and isMirrorTree(tree1.right, tree2.left)
            )

        if not root:
            return True
        return isMirrorTree(root.left, root.right)

    # 687*
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        self.pathLength = 0

        def DFS(node):
            if not node:
                return 0
            leftUnivalueHeight, rightUnivalueHeight = DFS(node.left), DFS(node.right)
            leftUnivalueHeight = leftUnivalueHeight + 1 if node.left and node.left.val == node.val else 0
            rightUnivalueHeight = rightUnivalueHeight + 1 if node.right and node.right.val == node.val else 0
            self.pathLength = max(self.pathLength, leftUnivalueHeight + rightUnivalueHeight)
            return max(leftUnivalueHeight, rightUnivalueHeight)

        DFS(root)
        return self.pathLength

    @cache
    def rob(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        robWithoutRoot = self.rob(root.left) + self.rob(root.right)

        robWithRoot = root.val
        robWithRoot += self.rob(root.left.left) + self.rob(root.left.right) if root.left else 0
        robWithRoot += self.rob(root.right.left) + self.rob(root.right.right) if root.right else 0

        return max(robWithRoot, robWithoutRoot)

    # 669*
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        def trim(node):
            return self.trimBST(node, low, high)

        if root is None:
            return root
        if low > root.val:
            return trim(root.right)
        if high < root.val:
            return trim(root.left)
        root.left = trim(root.left)
        root.right = trim(root.right)
        return root

    # 235*
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is None:
            return right
        if right is None:
            return left
        return root

    # 105*
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if len(preorder) == 0:
            return None
        root = TreeNode(val=preorder[0])
        divider = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : divider + 1], inorder[:divider])
        root.right = self.buildTree(preorder[divider + 1 :], inorder[divider + 1 :])
        return root

    # 124*
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.ans = float("-inf")

        def DFS(node: Optional[TreeNode]):
            if node is None:
                return 0
            leftSum = DFS(node.left)
            rightSum = DFS(node.right)
            self.ans = max(self.ans, node.val + max(leftSum, 0) + max(rightSum, 0))
            return node.val + max(leftSum, rightSum, 0)

        DFS(root)
        return self.ans

    # B. BST
    # 230
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.count = 0

        def DFS(node):
            if node is None:
                return -1
            left = DFS(node.left)
            if left != -1:
                return left
            self.count += 1
            if self.count == k:
                return node.val
            right = DFS(node.right)
            if right != -1:
                return right
            return -1

        return DFS(root)

    # 538
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.curSum = 0

        def DFS(node):
            if not node:
                return
            DFS(node.right)
            node.val += self.curSum
            self.curSum = node.val
            DFS(node.left)

        DFS(root)
        return root

    # 108*
    # O(nlog(n))
    # def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    #     if len(nums) == 0:
    #         return None
    #     mid = len(nums) // 2
    #     return TreeNode(
    #         val=nums[mid], left=self.sortedArrayToBST(nums[:mid]), right=self.sortedArrayToBST(nums[mid + 1 :])
    #     )
    # This solution creates new lists for nums[:mid] and nums[mid + 1 :] and hence has complexity O(nlog(n))
    # O(n)
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def toBSTHelper(start, end):
            if start > end:
                return None
            mid = (start + end) // 2
            return TreeNode(val=nums[mid], left=toBSTHelper(start, mid - 1), right=toBSTHelper(mid + 1, end))

        return toBSTHelper(0, len(nums) - 1)

    # 109*
    # Without converting into an array, which is O(n) whereas this one is O(nlog(n))
    def findPreMid(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast, preMid = head, head, head
        while fast and fast.next:
            preMid = slow
            slow = slow.next
            fast = fast.next.next
        return preMid

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        preMid = self.findPreMid(head)
        if not preMid:
            return None
        mid = preMid.next
        if not mid:
            return TreeNode(val=preMid.val)
        preMid.next = None
        return TreeNode(val=mid.val, left=self.sortedListToBST(head), right=self.sortedListToBST(mid.next))

    # 653
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        self.nums = []

        def inOrder(node):
            if node is None:
                return
            inOrder(node.left)
            self.nums.append(node.val)
            inOrder(node.right)

        inOrder(root)

        low, high = 0, len(self.nums) - 1
        while low < high:
            if self.nums[low] + self.nums[high] < k:
                low += 1
            elif self.nums[low] + self.nums[high] > k:
                high -= 1
            else:
                return True
        return False

    # 530
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        self.minimumDifference, self.lastVal = float("inf"), None

        def inOrder(node):
            if node is None:
                return
            inOrder(node.left)
            if self.lastVal is not None:
                self.minimumDifference = min(self.minimumDifference, node.val - self.lastVal)
            self.lastVal = node.val
            inOrder(node.right)

        inOrder(root)
        return self.minimumDifference

    # 501
    # iterative method 1
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        result, mostFrequency = [], 0
        curVal, curFrequency = None, 0
        stack = [root]
        while stack:
            curNode = stack.pop()
            if curNode is None:
                continue
            stack.append(curNode.right)
            while curNode.left:
                stack.append(curNode.left)
                curNode = curNode.left

            if curNode.val == curVal:
                curFrequency += 1
            else:
                curFrequency = 1
                curVal = curNode.val
            if curFrequency > mostFrequency:
                mostFrequency = curFrequency
                result = [curNode.val]
            elif curFrequency == mostFrequency:
                result.append(curNode.val)
        return result

    # iterative method 2
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        result, mostFrequency = [], 0
        curVal, curFrequency = None, 0
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            if not stack:
                return result
            curNode = stack.pop()
            root = curNode.right

            if curNode.val == curVal:
                curFrequency += 1
            else:
                curFrequency = 1
                curVal = curNode.val
            if curFrequency > mostFrequency:
                mostFrequency = curFrequency
                result = [curNode.val]
            elif curFrequency == mostFrequency:
                result.append(curNode.val)

    # C. TRIE
    # 208


class TrieNode:
    def __init__(self):
        self.word = False
        self.children = {}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for i in word:
            if i not in node.children:
                node.children[i] = TrieNode()
            node = node.children[i]
        node.word = True

    def search(self, word: str) -> bool:
        node = self.root
        for i in word:
            if i in node.children:
                node = node.children[i]
            else:
                return False
        return node.word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for i in prefix:
            if i in node.children:
                node = node.children[i]
            else:
                return False
        return True

    # 212


class TrieNode:
    def __init__(self, parent=None):
        self.children = {}
        self.isEnd = False
        self.parent = parent

    def addWord(self, word: str):
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode(cur)
            cur = cur.children[c]
        cur.isEnd = True

    def removeWord(self, word):
        self.isEnd = False
        cur = self
        while cur and word:
            if len(cur.children) > 0:
                break
            else:
                cur = cur.parent
                last, word = word[-1], word[:-1]
                del cur.children[last]

    # 677


class MapSum:
    def __init__(self):
        self.trie = self.TrieNode()
        self.wordMap = {}

    class TrieNode:
        def __init__(self):
            self.val = 0
            self.children = {}

    def insert(self, key: str, val: int) -> None:
        diff = val - self.wordMap.get(key, 0)
        self.wordMap[key] = val

        curTrieNode = self.trie
        curTrieNode.val += diff
        for letter in key:
            if letter not in curTrieNode.children:
                curTrieNode.children[letter] = self.TrieNode()
            curTrieNode = curTrieNode.children[letter]
            curTrieNode.val += diff

    def sum(self, prefix: str) -> int:
        curTrieNode = self.trie
        for letter in prefix:
            if letter not in curTrieNode.children:
                return 0
            curTrieNode = curTrieNode.children[letter]
        return curTrieNode.val
