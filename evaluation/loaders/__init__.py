"""Dataset loaders for VLM evaluation."""
from .diagram_mcq import DiagramMCQDataset
from .chartqa_custom import (
    ChartQACustomDataset,
    ChartQAWithCSV,
    ChartQAWithAnnotations,
    ChartQAWithBoth
)
from .clevr import (
    CLEVRDataset,
    CLEVRTestDataset,
    CLEVRImageOnlyDataset,
    CLEVRImageTextDataset
)
from .arc_agi import (
    ARCAGIDataset,
    ARCAGIReasoningDataset,
    ARCAGITrainingDataset
)
from .gui_360 import (
    GUI360Dataset,
    GUI360VisualDataset,
    GUI360A11yDataset,
    GUI360ExcelDataset,
    GUI360WordDataset,
    GUI360PPTDataset,
)
from .chartinsights import (
    ChartInsightsDataset,
    ChartInsightsOverallMCQ,
    ChartInsightsOverallMCQWithCSV,
    ChartInsightsOverallJudgement,
    ChartInsightsOverallFillBlank,
    ChartInsightsTextualMCQ,
    ChartInsightsTextualMCQWithCSV,
)

__all__ = [
    'DiagramMCQDataset',
    'ChartQACustomDataset',
    'ChartQAWithCSV',
    'ChartQAWithAnnotations',
    'ChartQAWithBoth',
    'CLEVRDataset',
    'CLEVRTestDataset',
    'CLEVRImageOnlyDataset',
    'CLEVRImageTextDataset',
    'ARCAGIDataset',
    'ARCAGIReasoningDataset',
    'ARCAGITrainingDataset',
    'GUI360Dataset',
    'GUI360VisualDataset',
    'GUI360A11yDataset',
    'GUI360ExcelDataset',
    'GUI360WordDataset',
    'GUI360PPTDataset',
    'ChartInsightsDataset',
    'ChartInsightsOverallMCQ',
    'ChartInsightsOverallMCQWithCSV',
    'ChartInsightsOverallJudgement',
    'ChartInsightsOverallFillBlank',
    'ChartInsightsTextualMCQ',
    'ChartInsightsTextualMCQWithCSV',
]
