"""RecurQuant recurrent-state quantization primitives."""

from .evidence import verify_evidence_artifact
from .finite_difference import (
    DirectionalDerivativeCheck,
    FiniteDifferencePoint,
    check_directional_derivative,
)
from .fisher_sensitivity import (
    GDNInt4TrajectorySensitivityCalibrator,
    GDNOneStepSensitivityCalibrator,
    RowPromotionSensitivityScores,
    SensitivityStepResult,
    TaskMacroSensitivityAccumulator,
    TaskMacroSensitivitySummary,
    row_promotion_scores_from_errors,
    row_promotion_sensitivity_scores,
)
from .horizon import (
    HorizonReadRisk,
    finite_horizon_row_read_risk,
    finite_horizon_row_read_risk_from_energies,
)
from .horizon_calibration import (
    BitwidthHorizonReadRisk,
    GDNCalibrationTrace,
    GDNHorizonCalibrationRecorder,
    TaskMacroHorizonAccumulator,
    TaskMacroHorizonSummary,
    row_quantization_error_energies,
    score_gdn_calibration_trace,
)
from .intervention import (
    PhysicalMetricRun,
    PhysicalRowPromotionOracleResult,
    RowPromotionMeasurement,
    evaluate_physical_row_promotions,
    target_nll_values,
)
from .mixed_quantization import PackedMixedQuantizedTensor, quantize_pack_mixed
from .model_fisher import (
    RowBlockModelFisherRisk,
    row_block_model_fisher_risk,
    sample_model_pseudo_labels,
)
from .packed_cache import (
    AdaptiveMixedPackedRecurrentStateCache,
    CoraMixedPackedLinearAttentionLayer,
    CoraMixedPackedRecurrentStateCache,
    MixedPackedRecurrentStateCache,
    PackedRecurrentStateCache,
    QueryEmaMixedPackedLinearAttentionLayer,
    QueryEmaMixedPackedRecurrentStateCache,
    RankFusedMixedPackedRecurrentStateCache,
)
from .quantization import (
    PackedQuantizedTensor,
    QuantizationResult,
    QuantizationSpec,
    quantize_dequantize,
    quantize_pack,
)
from .query_energy import Qwen35QueryEnergyObserver
from .qwen35 import (
    create_qwen35_adaptive_exact_budget_cache,
    create_qwen35_cora_exact_budget_cache,
    create_qwen35_exact_budget_cache,
    create_qwen35_packed_cache,
    create_qwen35_query_ema_exact_budget_cache,
    create_qwen35_rank_fused_exact_budget_cache,
    create_qwen35_v02_mixed_cache,
)
from .row_policy import ExactBudgetRowPlan, RowLocation, select_rows_exact_budget
from .transition_observer import Qwen35TransitionObserver

__all__ = [
    "AdaptiveMixedPackedRecurrentStateCache",
    "BitwidthHorizonReadRisk",
    "CoraMixedPackedLinearAttentionLayer",
    "CoraMixedPackedRecurrentStateCache",
    "DirectionalDerivativeCheck",
    "ExactBudgetRowPlan",
    "FiniteDifferencePoint",
    "GDNCalibrationTrace",
    "GDNHorizonCalibrationRecorder",
    "GDNInt4TrajectorySensitivityCalibrator",
    "GDNOneStepSensitivityCalibrator",
    "HorizonReadRisk",
    "MixedPackedRecurrentStateCache",
    "PackedMixedQuantizedTensor",
    "PackedQuantizedTensor",
    "PackedRecurrentStateCache",
    "QueryEmaMixedPackedLinearAttentionLayer",
    "QueryEmaMixedPackedRecurrentStateCache",
    "Qwen35QueryEnergyObserver",
    "Qwen35TransitionObserver",
    "RankFusedMixedPackedRecurrentStateCache",
    "PhysicalMetricRun",
    "PhysicalRowPromotionOracleResult",
    "QuantizationResult",
    "QuantizationSpec",
    "RowBlockModelFisherRisk",
    "RowLocation",
    "RowPromotionMeasurement",
    "RowPromotionSensitivityScores",
    "SensitivityStepResult",
    "TaskMacroHorizonAccumulator",
    "TaskMacroHorizonSummary",
    "TaskMacroSensitivityAccumulator",
    "TaskMacroSensitivitySummary",
    "check_directional_derivative",
    "create_qwen35_adaptive_exact_budget_cache",
    "create_qwen35_cora_exact_budget_cache",
    "create_qwen35_exact_budget_cache",
    "create_qwen35_packed_cache",
    "create_qwen35_query_ema_exact_budget_cache",
    "create_qwen35_rank_fused_exact_budget_cache",
    "create_qwen35_v02_mixed_cache",
    "finite_horizon_row_read_risk",
    "finite_horizon_row_read_risk_from_energies",
    "evaluate_physical_row_promotions",
    "quantize_dequantize",
    "quantize_pack",
    "quantize_pack_mixed",
    "row_block_model_fisher_risk",
    "row_promotion_scores_from_errors",
    "row_promotion_sensitivity_scores",
    "row_quantization_error_energies",
    "sample_model_pseudo_labels",
    "score_gdn_calibration_trace",
    "select_rows_exact_budget",
    "target_nll_values",
    "verify_evidence_artifact",
]

__version__ = "0.2.0a1"
