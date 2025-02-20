def two_sum(nums, target):
    # Create a dictionary to hold the numbers and their indices
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    # According to the problem statement, there is always exactly one solution.
    return []


# Test cases
if __name__ == '__main__':
    # Test case 1
    nums1 = [2, 7, 11, 15]
    target1 = 9
    expected1 = [0, 1]  # 2 + 7 = 9
    result1 = two_sum(nums1, target1)
    print("Test case 1:\nInput: nums = {} target = {}\nOutput: {}\nExpected: {}\n".format(nums1, target1, result1, expected1))

    # Test case 2
    nums2 = [3, 2, 4]
    target2 = 6
    expected2 = [1, 2]  # 2 + 4 = 6
    result2 = two_sum(nums2, target2)
    print("Test case 2:\nInput: nums = {} target = {}\nOutput: {}\nExpected: {}\n".format(nums2, target2, result2, expected2))

    # Test case 3
    nums3 = [3, 3]
    target3 = 6
    expected3 = [0, 1]  # 3 + 3 = 6
    result3 = two_sum(nums3, target3)
    print("Test case 3:\nInput: nums = {} target = {}\nOutput: {}\nExpected: {}\n".format(nums3, target3, result3, expected3))
