from devtools import debug

from src.data import eval_challenges, training_challenges
from src.main import solve_challenge
from src.models import (
    Attempt,
    AttemptEdge,
    FixAttemptConfig,
    FixPromptConfig,
    KTopConfig,
    LLMConfig,
    Model,
    PoolingConfig,
    Prompt,
    PromptConfig,
    RootAttemptConfig,
    RootPromptConfig,
)

# challenge = eval_challenges["963f59bc"]
# challenge = eval_challenges["d47aa2ff"]
# challenge = eval_challenges["00576224"]
# challenge = eval_challenges["009d5c81"]
# challenge = eval_challenges["0a2355a6"]
# challenge = eval_challenges["e5c44e8f"]
# challenge = eval_challenges["b457fec5"]
# challenge = eval_challenges["5b692c0f"]
# challenge = eval_challenges["256b0a75"]
# challenge = eval_challenges["f9d67f8b"]
# challenge = eval_challenges["2546ccf6"]
# challenge = eval_challenges["b4a43f3b"]
# challenge = eval_challenges["bf32578f"]
challenge = eval_challenges["bd14c3bf"]

# content = content_from_challenge(
#     challenge=eval_challenges["963f59bc"], include_diffs=True
# )
# debug(content)
"""
debug(
    challenge_to_messages(
        challenge=challenge,
        add_examples=False,
        prompt=Prompt.ONLY_GRID,
        include_diffs=False,
        include_image=False,
        use_ascii_over_array=False,
    )
)
"""

test_attempt_config_grid_only = RootAttemptConfig(
    attempts=50,
    llm_config=LLMConfig(
        model=Model.claude_3_5_sonnet,
        temperature=0.95,
    ),
    prompt_config=RootPromptConfig(
        base_prompt=Prompt.ONLY_GRID,
        use_examples=False,
        use_diffs=False,
        use_images=True,
        use_array=True,
        use_ascii=True,
        use_image=True,
    ),
    fixes=[
        AttemptEdge(
            k_top_config=KTopConfig(k_top=0, unique_code=False, unique_output=False),
            configs=[
                FixAttemptConfig(
                    attempts=2,
                    llm_config=LLMConfig(
                        model=Model.claude_3_5_sonnet, temperature=0.95
                    ),
                    prompt_config=FixPromptConfig(
                        base_prompt=Prompt.ONLY_GRID,
                        use_array=True,
                        use_ascii=True,
                        use_image=True,
                        use_fix_reasoning_tags=True,
                        use_fix_fail_line=True,
                        use_typical_issue_text=True,
                        include_diffs=True,
                    ),
                    fixes=[],
                )
            ],
        )
    ],
)
test_attempt_config_cot_test = RootAttemptConfig(
    attempts=100,
    llm_config=LLMConfig(model=Model.claude_3_5_sonnet, temperature=0.95),
    prompt_config=RootPromptConfig(
        base_prompt=Prompt.COT,
        use_examples=True,
        use_diffs=True,
        use_images=True,
        use_array=True,
        use_ascii=True,
        use_image=True,
    ),
    fixes=[],
)
test_attempt_config_reasoning_test = RootAttemptConfig(
    attempts=100,
    llm_config=LLMConfig(model=Model.claude_3_5_sonnet, temperature=0.95),
    prompt_config=RootPromptConfig(
        base_prompt=Prompt.REASONING,
        use_examples=True,
        use_diffs=True,
        use_images=True,
        use_array=True,
        use_ascii=True,
        use_image=True,
    ),
    fixes=[
        AttemptEdge(
            pooling=PoolingConfig(size=10),
            k_top_config=KTopConfig(k_top=10, unique_code=False, unique_output=False),
            configs=[
                FixAttemptConfig(
                    attempts=3,
                    llm_config=LLMConfig(
                        model=Model.claude_3_5_sonnet, temperature=0.95
                    ),
                    prompt_config=FixPromptConfig(
                        base_prompt=Prompt.REASONING,
                        use_array=True,
                        use_ascii=True,
                        use_image=True,
                        use_fix_reasoning_tags=True,
                        use_fix_fail_line=True,
                        use_typical_issue_text=True,
                        include_diffs=True,
                    ),
                    fixes=[
                        AttemptEdge(
                            pooling=PoolingConfig(size=3),
                            k_top_config=KTopConfig(
                                k_top=10, unique_code=False, unique_output=False
                            ),
                            configs=[
                                FixAttemptConfig(
                                    attempts=3,
                                    llm_config=LLMConfig(
                                        model=Model.claude_3_5_sonnet, temperature=0.95
                                    ),
                                    prompt_config=FixPromptConfig(
                                        base_prompt=Prompt.REASONING,
                                        use_array=True,
                                        use_ascii=True,
                                        use_image=True,
                                        use_fix_reasoning_tags=True,
                                        use_fix_fail_line=True,
                                        use_typical_issue_text=True,
                                        include_diffs=True,
                                    ),
                                    fixes=[
                                        AttemptEdge(
                                            pooling=PoolingConfig(size=3),
                                            k_top_config=KTopConfig(
                                                k_top=10,
                                                unique_code=False,
                                                unique_output=False,
                                            ),
                                            configs=[
                                                FixAttemptConfig(
                                                    attempts=3,
                                                    llm_config=LLMConfig(
                                                        model=Model.claude_3_5_sonnet,
                                                        temperature=0.95,
                                                    ),
                                                    prompt_config=FixPromptConfig(
                                                        base_prompt=Prompt.REASONING,
                                                        use_array=True,
                                                        use_ascii=True,
                                                        use_image=True,
                                                        use_fix_reasoning_tags=True,
                                                        use_fix_fail_line=True,
                                                        use_typical_issue_text=True,
                                                        include_diffs=True,
                                                    ),
                                                    fixes=[
                                                        AttemptEdge(
                                                            pooling=PoolingConfig(
                                                                size=3
                                                            ),
                                                            k_top_config=KTopConfig(
                                                                k_top=10,
                                                                unique_code=False,
                                                                unique_output=False,
                                                            ),
                                                            configs=[
                                                                FixAttemptConfig(
                                                                    attempts=10,
                                                                    llm_config=LLMConfig(
                                                                        model=Model.claude_3_5_sonnet,
                                                                        temperature=0.95,
                                                                    ),
                                                                    prompt_config=FixPromptConfig(
                                                                        base_prompt=Prompt.REASONING,
                                                                        use_array=True,
                                                                        use_ascii=True,
                                                                        use_image=True,
                                                                        use_fix_reasoning_tags=True,
                                                                        use_fix_fail_line=True,
                                                                        use_typical_issue_text=True,
                                                                        include_diffs=True,
                                                                    ),
                                                                    fixes=[
                                                                        AttemptEdge(
                                                                            pooling=PoolingConfig(
                                                                                size=3
                                                                            ),
                                                                            k_top_config=KTopConfig(
                                                                                k_top=10,
                                                                                unique_code=False,
                                                                                unique_output=False,
                                                                            ),
                                                                            configs=[
                                                                                FixAttemptConfig(
                                                                                    attempts=3,
                                                                                    llm_config=LLMConfig(
                                                                                        model=Model.claude_3_5_sonnet,
                                                                                        temperature=0.95,
                                                                                    ),
                                                                                    prompt_config=FixPromptConfig(
                                                                                        base_prompt=Prompt.REASONING,
                                                                                        use_array=True,
                                                                                        use_ascii=True,
                                                                                        use_image=True,
                                                                                        use_fix_reasoning_tags=True,
                                                                                        use_fix_fail_line=True,
                                                                                        use_typical_issue_text=True,
                                                                                        include_diffs=True,
                                                                                    ),
                                                                                    fixes=[],
                                                                                )
                                                                            ],
                                                                        )
                                                                    ],
                                                                )
                                                            ],
                                                        )
                                                    ],
                                                )
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    ],
)
test_attempt_config_reasoning = RootAttemptConfig(
    attempts=25,
    llm_config=LLMConfig(model=Model.claude_3_5_sonnet, temperature=0.95),
    prompt_config=RootPromptConfig(
        base_prompt=Prompt.REASONING,
        use_examples=True,
        use_diffs=True,
        use_images=True,
        use_array=True,
        use_ascii=True,
        use_image=True,
    ),
    fixes=[
        AttemptEdge(
            k_top_config=KTopConfig(k_top=10, unique_code=False, unique_output=False),
            configs=[
                FixAttemptConfig(
                    attempts=1,
                    llm_config=LLMConfig(
                        model=Model.claude_3_5_sonnet, temperature=0.95
                    ),
                    prompt_config=FixPromptConfig(
                        base_prompt=Prompt.REASONING,
                        use_fix_reasoning_tags=True,
                        use_fix_fail_line=True,
                        use_typical_issue_text=True,
                        include_diffs=True,
                        use_array=True,
                        use_ascii=True,
                        use_image=True,
                    ),
                    fixes=[],
                )
            ],
        )
    ],
)


async def main() -> None:
    # tree = [test_attempt_config_reasoning]
    # tree = [test_attempt_config_grid_only]
    # tree = [test_attempt_config_cot_test]
    # tree = [test_attempt_config_reasoning_test]
    tree = [test_attempt_config_grid_only]
    await solve_challenge(tree=tree, challenge=challenge)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# GET THIS TO WORK AND SEE HOW IT DOES WHEN YOU DO IT 1k times... will also be cheap with caching...
# then ask it to explain and use those explanations to improve and check for correctness...
# but number 1 important thing to test is, is this approach capable of actually getting correct answers a lot of the time!!
