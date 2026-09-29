"""
Calculation Engine Tool.

Performs financial calculations including DCF valuation, financial ratios,
growth rate computation, and statistical analysis. All calculations are
performed locally with no external API dependency.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class CalculationEngineTool(BaseTool):
    """Perform financial calculations including DCF, ratios, and growth rates.

    Use this tool when you need to compute financial metrics from raw data,
    perform DCF valuations, calculate growth rates, or run statistical analysis.
    Returns the calculation result with all intermediate steps shown.
    """

    name = "calculation_engine"
    description = (
        "Performs financial calculations including DCF valuation, financial ratio "
        "computation, growth rate calculation, CAGR, WACC, and basic statistical "
        "analysis. Use this tool when you need to derive metrics from raw financial "
        "data rather than retrieving pre-computed figures. Returns results with "
        "intermediate calculation steps."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "calculation_type": {
                "type": "string",
                "enum": [
                    "dcf",
                    "ratio",
                    "growth_rate",
                    "cagr",
                    "wacc",
                    "margin",
                    "yoy_change",
                    "statistics",
                    "custom",
                ],
                "description": "Type of financial calculation to perform",
            },
            "inputs": {
                "type": "object",
                "description": (
                    "Input values for the calculation. Structure depends on "
                    "calculation_type. E.g., for 'ratio': {numerator, denominator, name}. "
                    "For 'dcf': {cash_flows, discount_rate, terminal_growth_rate}. "
                    "For 'cagr': {beginning_value, ending_value, periods}."
                ),
            },
        },
        "required": ["calculation_type", "inputs"],
    }
    fallback_tools = []  # No fallback — internal tool

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate calculation method."""
        calc_type = kwargs["calculation_type"]
        inputs = kwargs["inputs"]

        calculators = {
            "dcf": self._calc_dcf,
            "ratio": self._calc_ratio,
            "growth_rate": self._calc_growth_rate,
            "cagr": self._calc_cagr,
            "wacc": self._calc_wacc,
            "margin": self._calc_margin,
            "yoy_change": self._calc_yoy_change,
            "statistics": self._calc_statistics,
            "custom": self._calc_custom,
        }

        calculator = calculators.get(calc_type)
        if not calculator:
            return ToolResult(
                success=False,
                error=f"Unknown calculation type: '{calc_type}'",
                source=self.name,
            )

        try:
            result = calculator(inputs)
            return ToolResult(
                success=True,
                data=result,
                source=self.name,
            )
        except (ValueError, TypeError, ZeroDivisionError, KeyError) as e:
            return ToolResult(
                success=False,
                error=f"Calculation error ({calc_type}): {type(e).__name__}: {e}",
                source=self.name,
            )

    def _calc_dcf(self, inputs: dict) -> dict:
        """Discounted Cash Flow valuation.

        Required inputs:
            cash_flows: list of projected free cash flows
            discount_rate: WACC or required rate of return (e.g., 0.10 for 10%)
            terminal_growth_rate: long-term growth rate (e.g., 0.025 for 2.5%)
        Optional:
            shares_outstanding: to compute per-share value
        """
        cash_flows = inputs["cash_flows"]
        discount_rate = inputs["discount_rate"]
        terminal_growth = inputs["terminal_growth_rate"]
        shares = inputs.get("shares_outstanding")

        steps = []

        # PV of projected cash flows
        pv_cash_flows = []
        for i, cf in enumerate(cash_flows, 1):
            pv = cf / ((1 + discount_rate) ** i)
            pv_cash_flows.append(round(pv, 2))
            steps.append(f"Year {i}: CF={cf:,.0f}, PV={pv:,.2f}")

        total_pv_cf = sum(pv_cash_flows)
        steps.append(f"Total PV of projected cash flows: {total_pv_cf:,.2f}")

        # Terminal value (Gordon Growth Model)
        last_cf = cash_flows[-1]
        terminal_value = (last_cf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** len(cash_flows))
        steps.append(f"Terminal value: {terminal_value:,.2f}")
        steps.append(f"PV of terminal value: {pv_terminal:,.2f}")

        # Enterprise value
        enterprise_value = total_pv_cf + pv_terminal
        steps.append(f"Enterprise value: {enterprise_value:,.2f}")

        result = {
            "calculation_type": "dcf",
            "enterprise_value": round(enterprise_value, 2),
            "pv_cash_flows": round(total_pv_cf, 2),
            "pv_terminal_value": round(pv_terminal, 2),
            "terminal_value": round(terminal_value, 2),
            "steps": steps,
        }

        if shares:
            per_share = enterprise_value / shares
            result["per_share_value"] = round(per_share, 2)
            result["shares_outstanding"] = shares
            steps.append(f"Per-share value: {per_share:,.2f}")

        return result

    def _calc_ratio(self, inputs: dict) -> dict:
        """Calculate a financial ratio."""
        numerator = inputs["numerator"]
        denominator = inputs["denominator"]
        name = inputs.get("name", "ratio")

        if denominator == 0:
            return {
                "calculation_type": "ratio",
                "name": name,
                "result": None,
                "error": "Division by zero — denominator is 0",
                "steps": [f"{name} = {numerator} / {denominator} → undefined"],
            }

        result = numerator / denominator
        return {
            "calculation_type": "ratio",
            "name": name,
            "result": round(result, 4),
            "steps": [f"{name} = {numerator:,.2f} / {denominator:,.2f} = {result:.4f}"],
        }

    def _calc_growth_rate(self, inputs: dict) -> dict:
        """Calculate period-over-period growth rate."""
        old_value = inputs["old_value"]
        new_value = inputs["new_value"]
        name = inputs.get("name", "growth_rate")

        if old_value == 0:
            return {
                "calculation_type": "growth_rate",
                "name": name,
                "result": None,
                "error": "Cannot compute growth rate from zero base",
            }

        rate = (new_value - old_value) / old_value
        return {
            "calculation_type": "growth_rate",
            "name": name,
            "result": round(rate, 4),
            "result_percentage": f"{rate * 100:.2f}%",
            "steps": [
                f"{name} = ({new_value:,.2f} - {old_value:,.2f}) / {old_value:,.2f}",
                f"= {rate:.4f} ({rate * 100:.2f}%)",
            ],
        }

    def _calc_cagr(self, inputs: dict) -> dict:
        """Compound Annual Growth Rate."""
        beg = inputs["beginning_value"]
        end = inputs["ending_value"]
        periods = inputs["periods"]

        if beg <= 0 or end <= 0:
            return {
                "calculation_type": "cagr",
                "result": None,
                "error": "CAGR requires positive beginning and ending values",
            }

        cagr = (end / beg) ** (1 / periods) - 1
        return {
            "calculation_type": "cagr",
            "result": round(cagr, 4),
            "result_percentage": f"{cagr * 100:.2f}%",
            "steps": [
                f"CAGR = ({end:,.2f} / {beg:,.2f})^(1/{periods}) - 1",
                f"= {cagr:.4f} ({cagr * 100:.2f}%)",
            ],
        }

    def _calc_wacc(self, inputs: dict) -> dict:
        """Weighted Average Cost of Capital."""
        equity = inputs["equity_value"]
        debt = inputs["debt_value"]
        cost_equity = inputs["cost_of_equity"]
        cost_debt = inputs["cost_of_debt"]
        tax_rate = inputs["tax_rate"]

        total = equity + debt
        weight_e = equity / total
        weight_d = debt / total
        wacc = (weight_e * cost_equity) + (weight_d * cost_debt * (1 - tax_rate))

        return {
            "calculation_type": "wacc",
            "result": round(wacc, 4),
            "result_percentage": f"{wacc * 100:.2f}%",
            "weight_equity": round(weight_e, 4),
            "weight_debt": round(weight_d, 4),
            "steps": [
                f"E/(E+D) = {equity:,.0f}/{total:,.0f} = {weight_e:.4f}",
                f"D/(E+D) = {debt:,.0f}/{total:,.0f} = {weight_d:.4f}",
                f"WACC = ({weight_e:.4f} × {cost_equity:.4f}) + ({weight_d:.4f} × {cost_debt:.4f} × (1 - {tax_rate:.2f}))",
                f"= {wacc:.4f} ({wacc * 100:.2f}%)",
            ],
        }

    def _calc_margin(self, inputs: dict) -> dict:
        """Calculate a margin (profit, operating, gross, etc.)."""
        amount = inputs["amount"]
        revenue = inputs["revenue"]
        name = inputs.get("name", "margin")

        if revenue == 0:
            return {"calculation_type": "margin", "name": name, "result": None, "error": "Revenue is zero"}

        margin = amount / revenue
        return {
            "calculation_type": "margin",
            "name": name,
            "result": round(margin, 4),
            "result_percentage": f"{margin * 100:.2f}%",
            "steps": [f"{name} = {amount:,.2f} / {revenue:,.2f} = {margin:.4f} ({margin * 100:.2f}%)"],
        }

    def _calc_yoy_change(self, inputs: dict) -> dict:
        """Year-over-year change for a series of values."""
        values = inputs["values"]
        labels = inputs.get("labels", [f"Period {i}" for i in range(len(values))])
        name = inputs.get("name", "YoY Change")

        changes = []
        for i in range(1, len(values)):
            if values[i - 1] == 0:
                changes.append({"period": labels[i], "change": None, "note": "base is zero"})
                continue
            chg = (values[i] - values[i - 1]) / values[i - 1]
            changes.append({
                "period": labels[i],
                "change": round(chg, 4),
                "change_pct": f"{chg * 100:.2f}%",
            })

        return {
            "calculation_type": "yoy_change",
            "name": name,
            "changes": changes,
        }

    def _calc_statistics(self, inputs: dict) -> dict:
        """Basic statistical analysis on a list of numbers."""
        values = inputs["values"]
        name = inputs.get("name", "statistics")

        n = len(values)
        if n == 0:
            return {"calculation_type": "statistics", "name": name, "error": "Empty dataset"}

        mean = sum(values) / n
        sorted_vals = sorted(values)
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)

        return {
            "calculation_type": "statistics",
            "name": name,
            "count": n,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std_dev": round(std_dev, 4),
            "variance": round(variance, 4),
        }

    def _calc_custom(self, inputs: dict) -> dict:
        """Evaluate a custom formula with provided variables.

        inputs:
            formula: str — Python math expression (e.g., "a / b * 100")
            variables: dict — variable values (e.g., {"a": 500, "b": 1000})
        """
        formula = inputs["formula"]
        variables = inputs.get("variables", {})

        # Safety: only allow basic math operations
        import re
        safe_pattern = r'^[\d\s\+\-\*/\(\)\.\,a-zA-Z_]+$'
        if not re.match(safe_pattern, formula):
            return {
                "calculation_type": "custom",
                "error": "Formula contains disallowed characters",
            }

        try:
            result = eval(formula, {"__builtins__": {}, "math": math}, variables)
            return {
                "calculation_type": "custom",
                "formula": formula,
                "variables": variables,
                "result": round(float(result), 4) if isinstance(result, (int, float)) else result,
            }
        except Exception as e:
            return {
                "calculation_type": "custom",
                "formula": formula,
                "error": f"Formula evaluation failed: {e}",
            }
