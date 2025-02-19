import asyncio
import random

from devtools import debug

from src.data import eval_challenges, training_challenges
from src.logic import solve_challenge


async def main() -> None:
    num_correct: int = 0
    num_tested: int = 0
    # 40 percent for first 10 in train w claude

    challenge_ids = [
        # "007bbfb7", # correct
        # "00d62c1b", # wrong
        # "017c7c7b", # correct
        # "025d127b", # wrong
        # "045e512c", # wrong
        # "0520fde7", # correct
        "05269061",  # wrong: technically got it wrong but it did have a correct solution with train attempt 1 correct
        # "05f2a901",  # correct
        # "06df4c85",  # wrong
        "08ed6ac7",  # wrong -> but right second time around
    ]

    random_challenge_ids = [
        # "2c608aff",  # correct
        # "3c9b0459",  # correct
        "4522001f",  # wrong
        # "50cb2852",  # correct
        # "7f4411dc",  # correct
    ]

    challenge_eval_ids = [
        "00576224",  # correct
        "009d5c81",  # correct
        "00dbd492",  # correct
        "03560426",  # wrong
        "05a7bcf2",  # wrong
    ]

    random_challenge_eval_ids = [
        "33b52de3",  # wrong
        "4aab4007",  # wrong
        "575b1a71",  # wrong
        "b7f8a4d8",  # wrong
        "e69241bd",  # correct
    ]

    # test eval inds 250, 251: 'a57f2f04' (wrong), 'a59b95c0' (correct) [50%]
    # 252:257: 'a680ac02' (wrong), 'a8610ef7' (wrong), 'a934301b' (wrong but almost, drill down on debugs more), 'aa18de87' (correct), 'aa300dc3' (wrong but close)
    # 252:257 [no examples]: 'a680ac02' (correct), 'a8610ef7' (wrong), 'a934301b' (wrong), 'aa18de87' (correct), 'aa300dc3' (wrong, got it correct a few times)
    # 257:260 [no examples]: 'aa4ec2a5' (wrong)

    # eval_ids_to_test = list(eval_challenges.keys())[272:274]

    # do random set of 10
    eval_keys = list(eval_challenges.keys())
    random.shuffle(eval_keys)
    eval_ids_to_test = eval_keys[:5]
    # take random 10 sample fro eval_ids_to_test
    debug(eval_ids_to_test)

    # 40% on this random set...
    eval_ids_to_test_miss = [
        # "e6de6e8f",  # wrong
        # "48f8583b",  # correct
        # "c64f1187",  # wrong and off by a lot
        # "b0f4d537",  # wrong and off by a lot -- USE THIS AS PROMPT CASE
        "319f2597",  # correct
        "5833af48",  # wrong
        "e95e3d8e",  # wrong
        "ccd554ac",  # correct
        "88207623",  # wrong
        "50a16a69",  # correct
    ]
    eval_ids_to_test_miss = [
        # "2a5f8217",  # correct
        # "0c786b71",  # correct
        # "66f2d22f",  # correct
        # "7bb29440", # correct
        # "aa18de87", # correct
        # "9a4bb226",  # correct
        # "e1d2900e", # correct
        # "414297c0", # wrong
        # "e0fb7511", # correct
        # "e760a62e",  # wrong
    ]

    # 5.5 out of 10
    eval_ids_to_test_miss = [
        # "c35c1b4c",  # wrong
        # "ac605cbb",  # wrong (x2)
        # "1d0a4b61", # correct
        # "ef26cbf6", # correct
        # "e9ac8c9e", # correct
        # "bf89d739",  # wrong (correct)
        # "b7fb29bc",  # wrong (x2)
        # "c48954c1",  # correct
        # "94be5b80",  # wrong
        # "27f8ce4f", # correct
    ]

    eval_ids_to_test_miss = [
        "ac3e2b04",  # wrong
        "929ab4e9",  # wrong (correct with no examples + w examples)
        # "cd3c21df", # correct
        # "516b51b7",  # correct
        # "00576224", # correct
        # "84f2aca1", # correct
        "696d4842",  # wrong
        "762cd429",  # wrong (but close)
        # "642248e4", # correct
        # "319f2597", # correct
    ]

    eval_ids_to_test_miss = [
        # "09c534e7", # wrong
        # "21f83797", # correct
        # "79369cc6", # wrong
        # "ff72ca3e", # correct
        # "5b692c0f",  # wrong
        # "b15fca0b",  # wrong
        # "aa4ec2a5", # wrong
        "68b67ca3",  # correct
        "b7fb29bc",  # wrong
        "963f59bc",  # wrong
    ]

    # more power worked for e7b06bea! When you are getting a solid number of examples correct but not all of them, double down!
    # eval_ids_to_test = ["ecaa0ec1"]  # TODO could benefit from reasoning

    eval_ids_to_test_miss = [
        # "94414823", # correct
        # "3ed85e70", # wrong
        # "17b80ad2", # correct
        # "af22c60d", # wrong
        # "ea959feb",
        # "981571dc",
        # "423a55dc", # correct
        # "d47aa2ff", # correct
        # "9356391f", # wrong
        # "7d419a02",
    ]

    eval_ids_to_test_miss = [
        # "5289ad53",
        "d47aa2ff"
    ]
    eval_ids_to_test_miss = [
        "aa18de87",
        "e133d23d",
        "e66aafb8",
        "72a961c9",
        "b457fec5",
    ]

    eval_ids_to_test_miss = [
        "0d87d2a6",  # wrong but all examples correct
        "5b692c0f",  # wrong
        # "903d1b4a",
        # "00dbd492",
        # "4cd1b7b2",
    ]

    eval_ids_to_test_miss = [
        "15696249",  # wrong but all examples correct
        "af24b4cc",  # correct
        "27a77e38",  # correct
        "1acc24af",  # wrong but many examples correct
        "e7dd8335",  # correct
    ]
    # so far, claude 60% (9/15)

    eval_ids_to_test_miss = [
        "a3f84088",  # correct but missed an example (got from fix)
        "256b0a75",  # wrong, no examples correct
        "45bbe264",  # correct
        "f9d67f8b",  # wrong, no examples correct, mask
        "cad67732",  # wrong, some examples correct
    ]
    # so far, claude 55% (11/20)
    eval_ids_to_test_miss = [
        "bf32578f",  # wrong GOOD FOR POOLING but some examples correct
        # "bbb1b8b6", # THIS MAKES AN ERROR, MUST FIX
        "25094a63",  # correct
        "a406ac07",  # correct
        "2c737e39",  # wrong but many times correct
    ]
    # so far, claude 54.16% (13/24)

    eval_ids_to_test_miss = [
        # 'ea9794b1', # correct from fix
        # 'f3e62deb', # ERROR AGAIN MUST FIX
        # "2546ccf6", # wrong, one example correct
        # "20818e16", # correct, after fixes!
        # "9caba7c3", # wrong
    ]
    # so far, claude 54.16% (15/28)
    eval_ids_to_test_miss = [
        "84f2aca1",  # correct
        "642248e4",  # correct
        "b4a43f3b",  # wrong # THIS WOULD BE GOOD EXAMPLE FOR TUNING..., requires IQ
        "7ee1c6ea",  # correct
        "ff72ca3e",  # correct
    ]
    # so far, claude 57.6% (19/33)

    eval_ids_to_test = [
        "bd14c3bf",  # wrong but some examples correct
        # "ca8de6ea",  # correct
        # "b0f4d537",
        # "54db823b",
        # "e69241bd",
    ]

    from src.trees import big_tree, medium_tree, small_tree
    from src.trees.small import (
        claude_flat_tree,
        claude_gemini_flat_tree,
        gemini_flat_tree,
        gemini_small_tree,
        gemini_tree,
        gpt_small_tree,
        haiku_gemini_flat_tree,
        o1_flat_tree,
        o1_mini_flat_tree,
        o1_mini_small_tree,
    )

    for challenge_id in eval_ids_to_test:
        debug(challenge_id)
        # challenge = training_challenges[challenge_id]
        challenge = eval_challenges[challenge_id]
        solutions = await solve_challenge(
            challenge=challenge,
            # tree=gemini_flat_tree,
            tree=claude_gemini_flat_tree,
            # tree=haiku_gemini_flat_tree,
        )
        test_output = challenge.test[0].output
        solution_one_correct = solutions[0] == test_output
        solution_two_correct = solutions[1] == test_output
        debug(solution_one_correct, solution_two_correct)
        is_correct_final = solution_one_correct or solution_two_correct
        debug(challenge_id, is_correct_final)
        if is_correct_final:
            num_correct = num_correct + 1
        num_tested = num_tested + 1
        print(f"Correct Percent SO FAR: {num_correct / num_tested}")

    print(f"Correct Percent: {num_correct / num_tested}")


if __name__ == "__main__":
    asyncio.run(main())
