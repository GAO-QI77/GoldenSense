from investment_output_policy import (
    find_directive_language,
    sanitize_research_language,
)


def test_policy_flags_directive_and_target_language():
    violations = find_directive_language(
        [
            "建议黄金目标暴露约 35%。",
            "现在可以买入并加仓。",
            "改写成持仓建议。",
        ]
    )

    assert {item["code"] for item in violations} == {
        "target_position",
        "trade_directive",
        "position_advice",
    }


def test_policy_rewrites_to_research_language():
    original = "建议黄金目标暴露约 35%，若突破则买入并加仓。"

    sanitized = sanitize_research_language(original)

    assert find_directive_language([sanitized]) == []
    assert "风险参考区间" in sanitized
    assert "关注上行情景" in sanitized
