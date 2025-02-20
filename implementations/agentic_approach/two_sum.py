def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in num_map:
            return [num_map[diff], i]
        num_map[num] = i
    return None

# Test case
def test_two_sum():
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    assert result == [0, 1], f"Expected [0, 1], got {result}"
    print("Test passed!")

if __name__ == "__main__":
    test_two_sum()
