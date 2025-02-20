def twoSum(nums, target):
    # Create a hash map to store value to index mapping
    hash_map = {}
    for i, num in enumerate(nums):
        diff = target - num
        # Check if the complement exists in the map
        if diff in hash_map:
            return [hash_map[diff], i]
        # Otherwise, store the current number and its index
        hash_map[num] = i
    return []

if __name__ == "__main__":
    # Test case 1:
    nums = [2, 7, 11, 15]
    target = 9
    print("Test case 1: nums = {} target = {}".format(nums, target))
    print("Output:", twoSum(nums, target))
    
    # Test case 2:
    nums = [3, 2, 4]
    target = 6
    print("Test case 2: nums = {} target = {}".format(nums, target))
    print("Output:", twoSum(nums, target))
    
    # Test case 3:
    nums = [3, 3]
    target = 6
    print("Test case 3: nums = {} target = {}".format(nums, target))
    print("Output:", twoSum(nums, target))
