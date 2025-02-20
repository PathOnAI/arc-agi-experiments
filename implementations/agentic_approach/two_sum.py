def two_sum(nums, target):
    # Dictionary to store the complement and its index
    complement = {}
    for idx, number in enumerate(nums):
        if target - number in complement:
            return [complement[target - number], idx]
        complement[number] = idx
    raise Exception('No two sum solution exists')


if __name__ == '__main__':
    # Test cases
    # Example 1:
    nums = [2, 7, 11, 15]
    target = 9
    expected = [0, 1]  # one possible answer
    result = two_sum(nums, target)
    print(f"Test case 1: nums={nums}, target={target}")
    print(f"Output: {result}")

    # Additional test cases
    # Example 2:
    nums = [3, 2, 4]
    target = 6
    expected = [1, 2]
    result = two_sum(nums, target)
    print(f"\nTest case 2: nums={nums}, target={target}")
    print(f"Output: {result}")

    # Example 3:
    nums = [3, 3]
    target = 6
    expected = [0, 1]
    result = two_sum(nums, target)
    print(f"\nTest case 3: nums={nums}, target={target}")
    print(f"Output: {result}")
