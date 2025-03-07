from src.models import (
    AttemptEdge,
    FixAttemptConfig,
    FixPromptConfig,
    KTopConfig,
    LLMConfig,
    Model,
    Prompt,
    RootAttemptConfig,
    RootPromptConfig,
)

medium_tree: list[RootAttemptConfig] = [
    RootAttemptConfig(
        attempts=20,
        llm_config=LLMConfig(
            model=Model.claude_3_5_sonnet,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[],
    ),
    RootAttemptConfig(
        attempts=200,
        llm_config=LLMConfig(
            model=Model.claude_3_5_sonnet,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[
            AttemptEdge(
                k_top_config=KTopConfig(
                    k_top=50, unique_code=True, unique_output=False
                ),
                configs=[
                    FixAttemptConfig(
                        attempts=1,
                        llm_config=LLMConfig(
                            model=Model.claude_3_5_sonnet,
                            temperature=0.95,
                        ),
                        prompt_config=FixPromptConfig(
                            base_prompt=Prompt.REASONING,
                            use_fix_reasoning_tags=True,
                            use_fix_fail_line=True,
                            use_typical_issue_text=True,
                            include_diffs=True,
                            use_ascii=True,
                            use_array=True,
                            use_image=True,
                        ),
                        fixes=[
                            AttemptEdge(
                                k_top_config=KTopConfig(
                                    k_top=10, unique_code=True, unique_output=False
                                ),
                                configs=[
                                    FixAttemptConfig(
                                        attempts=2,
                                        llm_config=LLMConfig(
                                            model=Model.claude_3_5_sonnet,
                                            temperature=0.95,
                                        ),
                                        prompt_config=FixPromptConfig(
                                            base_prompt=Prompt.REASONING,
                                            use_fix_reasoning_tags=True,
                                            use_fix_fail_line=True,
                                            use_typical_issue_text=True,
                                            include_diffs=True,
                                            use_ascii=True,
                                            use_array=True,
                                            use_image=True,
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
    ),
    RootAttemptConfig(
        attempts=1,
        llm_config=LLMConfig(
            model=Model.openrouter_o1_mini,
            temperature=0.95,
        ),
        prompt_config=RootPromptConfig(
            base_prompt=Prompt.REASONING,
            use_examples=True,
            use_diffs=True,
            use_images=True,
            use_ascii=True,
            use_array=True,
            use_image=True,
        ),
        fixes=[],
    ),
]
